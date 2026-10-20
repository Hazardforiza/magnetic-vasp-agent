"""Read-only monitoring and allowlisted result retrieval for remote VASP runs.

The public functions in this module accept typed cluster/run records, never a
remote command or an arbitrary remote path.  SSH is used only with a fixed
stdin script.  SFTP is restricted to exact, allowlisted basenames below.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

from magnetic_vasp_agent.execution_models import ClusterProfile
from magnetic_vasp_agent.execution_probe import SSH_OPTIONS
from magnetic_vasp_agent.execution_runtime_models import (
    LaunchBackend,
    RunRecord,
)
from magnetic_vasp_agent.transfer import SFTP_OPTIONS

Runner = Callable[..., object]

STATUS_FILES = (
    "OUTCAR",
    "OSZICAR",
    "vasprun.xml",
    "magvasp.stdout.log",
    "magvasp.stderr.log",
)
FETCH_ALLOWED_FILES = frozenset(
    {
        # Inputs needed to reproduce/post-process a run.  POTCAR is
        # deliberately excluded because its license does not permit copying.
        "INCAR",
        "KPOINTS",
        "POSCAR",
        "manifest.json",
        # VASP results and bounded diagnostic logs.
        "CONTCAR",
        "OUTCAR",
        "OSZICAR",
        "vasprun.xml",
        "DOSCAR",
        "EIGENVAL",
        "PROCAR",
        "CHGCAR",
        "CHG",
        "AECCAR0",
        "AECCAR1",
        "AECCAR2",
        "LOCPOT",
        "ELFCAR",
        "PARCHG",
        "IBZKPT",
        "XDATCAR",
        "REPORT",
        "magvasp.stdout.log",
        "magvasp.stderr.log",
        # Fixed VASPKIT products currently supported by the project.
        "band.dat",
        "BAND.dat",
        "TDOS.dat",
        "PDOS.dat",
        "KPATH.in",
        "KLABELS",
    }
)

MAX_STATUS_OUTPUT_BYTES = 512 * 1024
MAX_TAIL_BYTES = 64 * 1024
TAIL_LINES = 64
MAX_FETCH_FILE_BYTES = 2**31
MAX_FETCH_TOTAL_BYTES = 2**32
MAX_COMMAND_OUTPUT_BYTES = 2_000_000
MAX_FETCH_FILES = 32
STATUS_TIMEOUT_SECONDS = 30
HASH_TIMEOUT_SECONDS = 120
SFTP_TIMEOUT_SECONDS = 1800

_SAFE_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_SAFE_HOST = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,253}[A-Za-z0-9])?")
_SAFE_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExecutionMonitorError(ValueError):
    """Raised when a monitor/fetch operation cannot be performed safely."""


def _validated_context(
    profile: ClusterProfile, record: RunRecord
) -> tuple[ClusterProfile, RunRecord, str]:
    """Validate and bind one record to one configured remote run directory."""

    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    record = RunRecord.model_validate_json(record.model_dump_json())
    for label, value, pattern in (
        ("gateway", profile.gateway, _SAFE_HOST),
        ("gateway_hostname", profile.gateway_hostname, _SAFE_HOST),
        ("user", profile.user, _SAFE_USER),
    ):
        if pattern.fullmatch(value) is None or ".." in value:
            raise ExecutionMonitorError(
                f"SSH_IDENTITY_INVALID: {label} is not a fixed SSH token"
            )
    if not _record_hash_valid(record):
        raise ExecutionMonitorError(
            "RUN_RECORD_INTEGRITY: record_sha256 is missing or does not match"
        )
    if record.backend != LaunchBackend.REMOTE_MPI or record.cluster_execution != "remote_mpi":
        raise ExecutionMonitorError("REMOTE_RUN_REQUIRED: record is not a remote MPI run")
    if record.node is None or record.node == profile.gateway_hostname:
        raise ExecutionMonitorError("NODE_FORBIDDEN: record does not name a compute node")
    if record.node not in profile.allowed_nodes:
        raise ExecutionMonitorError("NODE_FORBIDDEN: node is not in the cluster profile")
    if record.remote_pid is None or not 2 <= record.remote_pid <= 2**31 - 1:
        raise ExecutionMonitorError("REMOTE_PID_INVALID: record has no safe remote PID")
    if profile.remote_root is None or record.remote_directory is None:
        raise ExecutionMonitorError("REMOTE_DIRECTORY_REQUIRED: profile/record is incomplete")

    root = _safe_remote_path(profile.remote_root, label="remote_root")
    workdir = _safe_remote_path(record.remote_directory, label="remote_directory")
    root_parts = PurePosixPath(root).parts
    work_parts = PurePosixPath(workdir).parts
    if work_parts[: len(root_parts)] != root_parts or len(work_parts) <= len(root_parts):
        raise ExecutionMonitorError(
            "REMOTE_DIRECTORY_SCOPE: run directory is outside the configured remote root"
        )
    # launch_preview stores runs at <root>/<workflow>--<run_id>/<step_id>.
    if len(work_parts) < len(root_parts) + 2 or not work_parts[-2].endswith(
        f"--{record.run_id}"
    ):
        raise ExecutionMonitorError(
            "REMOTE_DIRECTORY_BINDING: directory is not bound to the record run_id"
        )
    return profile, record, workdir


def _record_hash_valid(record: RunRecord) -> bool:
    payload = record.model_dump(mode="json")
    claimed = payload.pop("record_sha256", "")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    return bool(claimed) and hmac.compare_digest(claimed, expected)


def _safe_remote_path(value: str, *, label: str) -> str:
    if (
        not value.startswith("/")
        or value.endswith("/")
        or len(value) > 1024
        or "//" in value
    ):
        raise ExecutionMonitorError(f"{label.upper()}_INVALID: unsafe POSIX path")
    parts = value.split("/")[1:]
    if not parts or any(
        part in {".", ".."} or _SAFE_PART.fullmatch(part) is None for part in parts
    ):
        raise ExecutionMonitorError(f"{label.upper()}_INVALID: unsafe POSIX path")
    return value


def _ssh_argv(profile: ClusterProfile, node: str) -> list[str]:
    """Build the established gateway -> compute-node fixed SSH route."""

    gateway_script = (
        f'test "$(hostname -s)" = {shlex.quote(profile.gateway_hostname)} || exit 91\n'
        f'test "$(id -un)" = {shlex.quote(profile.user)} || exit 92\n'
        "exec "
        + shlex.join(
            [
                "ssh",
                *SSH_OPTIONS,
                "-p",
                str(profile.node_port),
                f"{profile.user}@{node}",
                "/bin/bash -s",
            ]
        )
    )
    return [
        "ssh",
        *SSH_OPTIONS,
        "-p",
        str(profile.port),
        f"{profile.user}@{profile.gateway}",
        "/bin/bash -c " + shlex.quote(gateway_script),
    ]


def _fixed_script(profile: ClusterProfile, node: str, body: str) -> bytes:
    return (
        "set -eu\n"
        "export LC_ALL=C\n"
        f'test "$(hostname -s)" = {shlex.quote(node)} || exit 91\n'
        f'test "$(id -un)" = {shlex.quote(profile.user)} || exit 92\n'
        f"{body}\n"
    ).encode("ascii")


def _directory_guard(workdir: str) -> str:
    quoted = shlex.quote(workdir)
    return (
        f"test -d {quoted} || exit 93\n"
        f"test ! -L {quoted} || exit 94\n"
        f'test "$(readlink -f -- {quoted})" = {quoted} || exit 94\n'
    )


def _status_body(workdir: str, pid: int) -> str:
    quoted = shlex.quote(workdir)
    lines = [
        _directory_guard(workdir),
        "printf 'MAGVASP_STATUS_V1\\n'",
        f"PID={pid}",
        "PID_ALIVE=no",
        "PID_CWD_MATCH=unknown",
        "if kill -0 \"$PID\" 2>/dev/null; then",
        "  PID_ALIVE=yes",
        "  PID_CWD=\"$(readlink -f -- /proc/$PID/cwd 2>/dev/null || true)\"",
        f"  if test \"$PID_CWD\" = {quoted}; then PID_CWD_MATCH=yes; else PID_CWD_MATCH=no; fi",
        # Recheck after /proc inspection so normal process exit is not reported
        # as a recycled/mismatched PID merely because it ended during the probe.
        "  if ! kill -0 \"$PID\" 2>/dev/null; then PID_ALIVE=no; PID_CWD_MATCH=unknown; fi",
        "fi",
        "printf 'pid_alive=%s\\n' \"$PID_ALIVE\"",
        "printf 'pid_cwd_match=%s\\n' \"$PID_CWD_MATCH\"",
    ]
    for name in STATUS_FILES:
        path = shlex.quote(f"{workdir}/{name}")
        lines.extend(
            [
                f"if test -L {path}; then",
                f"  printf 'file={name}|symlink|0\\n'",
                f"elif test -f {path}; then",
                f"  SIZE=\"$(stat -c %s -- {path})\"",
                f"  printf 'file={name}|regular|%s\\n' \"$SIZE\"",
                "else",
                f"  printf 'file={name}|missing|0\\n'",
                "fi",
            ]
        )
    for name in ("OUTCAR", "OSZICAR"):
        path = shlex.quote(f"{workdir}/{name}")
        lines.extend(
            [
                f"printf 'tail={name}|'",
                f"if test -f {path} && test ! -L {path}; then",
                (
                    f"  tail -n {TAIL_LINES} -- {path} | tail -c {MAX_TAIL_BYTES} "
                    "| base64 | tr -d '\\n'"
                ),
                "fi",
                "printf '\\n'",
            ]
        )
    lines.append("printf 'MAGVASP_STATUS_END\\n'")
    return "\n".join(lines)


def status_preview(profile: ClusterProfile, record: RunRecord) -> dict:
    """Return the fixed SSH invocation used for one read-only status probe."""

    profile, record, workdir = _validated_context(profile, record)
    return {
        "mode": "remote_status_preview",
        "run_id": record.run_id,
        "node": record.node,
        "remote_directory": workdir,
        "remote_pid": record.remote_pid,
        "ssh_argv": _ssh_argv(profile, record.node),
        "stdin_script": _fixed_script(
            profile, record.node, _status_body(workdir, record.remote_pid)
        ).decode("ascii"),
        "read_only": True,
    }


def _run_fixed(
    argv: list[str], stdin: bytes, *, timeout: int
) -> subprocess.CompletedProcess[bytes]:
    executable = shutil.which(argv[0])
    if executable is None:
        raise ExecutionMonitorError(
            f"{argv[0].upper()}_UNAVAILABLE: install native OpenSSH first"
        )
    try:
        result = subprocess.run(
            [executable, *argv[1:]],
            input=stdin,
            capture_output=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecutionMonitorError(
            f"{argv[0].upper()}_TIMEOUT: fixed remote operation timed out"
        ) from exc
    return result


def _result_bytes(result: object, name: str) -> bytes:
    value = getattr(result, name, b"") or b""
    if not isinstance(value, bytes):
        raise ExecutionMonitorError("RUNNER_PROTOCOL: stdout/stderr must be bytes")
    return value


def _execute(
    runner: Runner | None,
    argv: list[str],
    stdin: bytes,
    *,
    timeout: int,
    output_limit: int = MAX_COMMAND_OUTPUT_BYTES,
    label: str,
) -> bytes:
    try:
        result = (runner or _run_fixed)(list(argv), bytes(stdin), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ExecutionMonitorError(f"{label}_TIMEOUT: fixed remote operation timed out") from exc
    stdout = _result_bytes(result, "stdout")
    stderr = _result_bytes(result, "stderr")
    if len(stdout) + len(stderr) > output_limit:
        raise ExecutionMonitorError(f"{label}_OUTPUT_TOO_LARGE: refusing unbounded output")
    code = getattr(result, "returncode", None)
    if type(code) is not int:
        raise ExecutionMonitorError("RUNNER_PROTOCOL: returncode must be an integer")
    if code != 0:
        raise ExecutionMonitorError(f"{label}_FAILED: exit {code}")
    return stdout


def _parse_status(output: bytes) -> dict:
    try:
        text = output.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: output is not ASCII") from exc
    lines = text.splitlines()
    if not lines or lines[0] != "MAGVASP_STATUS_V1" or lines[-1] != "MAGVASP_STATUS_END":
        raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: missing status markers")

    alive: bool | None = None
    cwd_match: str | None = None
    files: dict[str, dict[str, int | bool]] = {}
    tails: dict[str, str] = {}
    for line in lines[1:-1]:
        if line.startswith("pid_alive="):
            if alive is not None or line[10:] not in {"yes", "no"}:
                raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: invalid pid status")
            alive = line[10:] == "yes"
        elif line.startswith("pid_cwd_match="):
            value = line[14:]
            if cwd_match is not None or value not in {"yes", "no", "unknown"}:
                raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: invalid pid identity")
            cwd_match = value
        elif line.startswith("file="):
            fields = line[5:].split("|")
            if len(fields) != 3 or fields[0] not in STATUS_FILES or fields[0] in files:
                raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: invalid file inventory")
            name, kind, size_text = fields
            if kind not in {"regular", "missing", "symlink"} or not size_text.isdigit():
                raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: invalid file metadata")
            size = _bounded_decimal(
                size_text,
                label="REMOTE_STATUS_PROTOCOL",
                maximum=2**63 - 1,
            )
            if kind != "regular" and size != 0:
                raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: invalid absent-file size")
            files[name] = {
                "exists": kind == "regular",
                "size": size,
                "symlink": kind == "symlink",
            }
        elif line.startswith("tail="):
            fields = line[5:].split("|", 1)
            if len(fields) != 2 or fields[0] not in {"OUTCAR", "OSZICAR"} or fields[0] in tails:
                raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: invalid tail record")
            try:
                raw = base64.b64decode(fields[1], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ExecutionMonitorError(
                    "REMOTE_STATUS_PROTOCOL: invalid tail encoding"
                ) from exc
            if len(raw) > MAX_TAIL_BYTES:
                raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: tail exceeds its byte limit")
            tails[fields[0]] = raw.decode("utf-8", errors="replace")
        else:
            raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: unexpected output")

    if (
        alive is None
        or cwd_match is None
        or set(files) != set(STATUS_FILES)
        or set(tails) != {"OUTCAR", "OSZICAR"}
    ):
        raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: incomplete status response")
    if (alive and cwd_match == "unknown") or (not alive and cwd_match != "unknown"):
        raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: inconsistent PID identity")
    if any(tails[name] and not files[name]["exists"] for name in tails):
        raise ExecutionMonitorError("REMOTE_STATUS_PROTOCOL: tail is not bound to a regular file")
    return {
        "pid_alive": alive,
        "pid_cwd_matches": cwd_match == "yes",
        "pid_cwd_observation": cwd_match,
        "files": files,
        "tails": tails,
    }


def remote_status(
    profile: ClusterProfile,
    record: RunRecord,
    *,
    runner: Runner | None = None,
) -> dict:
    """Read process/file status without asserting scientific convergence."""

    preview = status_preview(profile, record)
    stdout = _execute(
        runner,
        preview["ssh_argv"],
        preview["stdin_script"].encode("ascii"),
        timeout=STATUS_TIMEOUT_SECONDS,
        output_limit=MAX_STATUS_OUTPUT_BYTES,
        label="REMOTE_STATUS",
    )
    parsed = _parse_status(stdout)
    blockers: list[str] = []
    if parsed["pid_alive"] and parsed["pid_cwd_observation"] == "yes":
        state = "running"
    elif parsed["pid_alive"]:
        state = "failed"
        blockers.append("REMOTE_PID_IDENTITY_MISMATCH")
    elif (
        parsed["files"]["OUTCAR"]["exists"]
        and "General timing and accounting informations for this job:"
        in parsed["tails"]["OUTCAR"]
    ):
        state = "completed"
    else:
        state = "failed"
        blockers.append("CLEAN_COMPLETION_NOT_OBSERVED")
    if any(item["symlink"] for item in parsed["files"].values()):
        state = "failed"
        blockers.append("REMOTE_OUTPUT_SYMLINK")
    return {
        "schema_version": 1,
        "mode": "remote_status",
        "run_id": preview["run_id"],
        "node": preview["node"],
        "remote_directory": preview["remote_directory"],
        "remote_pid": preview["remote_pid"],
        "state": state,
        **parsed,
        "scientific_converged": None,
        "blockers": blockers,
        "limitations": [
            "completed means a stopped PID plus the normal VASP timing footer",
            "scientific convergence is not inferred from process or tail output",
        ],
    }


def _validated_names(filenames: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(filenames, Sequence) or isinstance(filenames, (str, bytes)):
        raise ExecutionMonitorError("FETCH_SCOPE: filenames must be a sequence of basenames")
    if not filenames or len(filenames) > MAX_FETCH_FILES:
        raise ExecutionMonitorError("FETCH_SCOPE: request must contain unique allowlisted files")
    names = tuple(filenames)
    if any(not isinstance(name, str) for name in names):
        raise ExecutionMonitorError("FETCH_SCOPE: filename is not allowlisted")
    if len(set(names)) != len(names):
        raise ExecutionMonitorError("FETCH_SCOPE: request must contain unique allowlisted files")
    for name in names:
        if name.upper() in {"POTCAR", "WAVECAR"}:
            raise ExecutionMonitorError(
                "FETCH_SENSITIVE_FILE: POTCAR and WAVECAR retrieval is forbidden"
            )
        if name not in FETCH_ALLOWED_FILES:
            raise ExecutionMonitorError("FETCH_SCOPE: filename is not allowlisted")
        if PurePosixPath(name).name != name or "/" in name or "\\" in name:
            raise ExecutionMonitorError("FETCH_SCOPE: paths and traversal are forbidden")
    return names


def _validated_destination(destination: Path) -> Path:
    destination = Path(destination)
    if not destination.is_absolute():
        raise ExecutionMonitorError("FETCH_DESTINATION: destination must be absolute")
    if ".." in destination.parts:
        raise ExecutionMonitorError("FETCH_DESTINATION: traversal is forbidden")
    if destination.exists() or destination.is_symlink():
        raise ExecutionMonitorError("FETCH_DESTINATION: destination must be new")
    parent = destination.parent
    if not parent.is_dir():
        raise ExecutionMonitorError("FETCH_DESTINATION: parent must be an existing real directory")
    for ancestor in (parent, *parent.parents):
        try:
            info = ancestor.lstat()
        except OSError as exc:
            raise ExecutionMonitorError(
                "FETCH_DESTINATION: parent ancestry cannot be inspected"
            ) from exc
        if _is_link_or_reparse(info):
            raise ExecutionMonitorError(
                "FETCH_DESTINATION: symlinked parent directories are forbidden"
            )
    resolved = destination.resolve(strict=False)
    if resolved.parent != parent.resolve(strict=True):
        raise ExecutionMonitorError("FETCH_DESTINATION: parent resolution changed")
    return resolved


def _inventory_body(workdir: str, names: tuple[str, ...]) -> str:
    lines = [
        _directory_guard(workdir),
        "printf 'MAGVASP_FETCH_V1\\n'",
        "TOTAL=0",
    ]
    for name in names:
        path = shlex.quote(f"{workdir}/{name}")
        lines.extend(
            [
                f"test ! -L {path} || exit 95",
                f"test -f {path} || exit 96",
                f"META=\"$(stat -c '%d:%i:%h:%s' -- {path})\"",
                'case "$META" in *:*:1:*) :;; *) exit 97;; esac',
                'SIZE="${META##*:}"',
                'case "$SIZE" in ""|*[!0-9]*) exit 97;; esac',
                f'test "$SIZE" -le {MAX_FETCH_FILE_BYTES} || exit 98',
                'TOTAL=$((TOTAL + SIZE))',
                f'test "$TOTAL" -le {MAX_FETCH_TOTAL_BYTES} || exit 99',
                f"HASH=\"$(sha256sum -- {path})\"",
                'HASH="${HASH%% *}"',
                f"test \"$META\" = \"$(stat -c '%d:%i:%h:%s' -- {path})\" || exit 100",
                f"printf 'file={name}|%s|%s\\n' \"$SIZE\" \"$HASH\"",
            ]
        )
    lines.append("printf 'MAGVASP_FETCH_END\\n'")
    return "\n".join(lines)


def _sftp_quote(value: str) -> str:
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ExecutionMonitorError("FETCH_PATH: invalid SFTP path")
    normalized = value.replace("\\", "/")
    return '"' + normalized.replace('"', '\\"') + '"'


def _sftp_argv(profile: ClusterProfile, node: str) -> list[str]:
    return [
        "sftp",
        *SFTP_OPTIONS,
        "-o",
        f"ProxyJump={profile.user}@{profile.gateway}:{profile.port}",
        "-P",
        str(profile.node_port),
        f"{profile.user}@{node}",
    ]


def _sftp_batch(workdir: str, destination: Path, names: tuple[str, ...]) -> str:
    lines = [
        f"get {_sftp_quote(workdir + '/' + name)} {_sftp_quote(str(destination / name))}"
        for name in names
    ]
    return "\n".join([*lines, "bye", ""])


def fetch_preview(
    profile: ClusterProfile,
    record: RunRecord,
    destination: Path,
    filenames: Sequence[str],
) -> dict:
    """Build fixed inventory/SFTP plans without creating the destination."""

    profile, record, workdir = _validated_context(profile, record)
    names = _validated_names(filenames)
    destination = _validated_destination(destination)
    inventory_script = _fixed_script(
        profile, record.node, _inventory_body(workdir, names)
    ).decode("ascii")
    return {
        "mode": "remote_fetch_preview",
        "run_id": record.run_id,
        "node": record.node,
        "remote_directory": workdir,
        "destination": str(destination),
        "files": list(names),
        "ssh_inventory_argv": _ssh_argv(profile, record.node),
        "ssh_inventory_stdin": inventory_script,
        "sftp_argv": _sftp_argv(profile, record.node),
        "sftp_batch": _sftp_batch(workdir, destination, names),
        "max_file_bytes": MAX_FETCH_FILE_BYTES,
        "max_total_bytes": MAX_FETCH_TOTAL_BYTES,
    }


def _parse_inventory(output: bytes, names: tuple[str, ...]) -> dict[str, dict[str, int | str]]:
    try:
        text = output.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ExecutionMonitorError("REMOTE_HASH_PROTOCOL: output is not ASCII") from exc
    lines = text.splitlines()
    if not lines or lines[0] != "MAGVASP_FETCH_V1" or lines[-1] != "MAGVASP_FETCH_END":
        raise ExecutionMonitorError("REMOTE_HASH_PROTOCOL: missing fetch markers")
    inventory: dict[str, dict[str, int | str]] = {}
    total = 0
    for line in lines[1:-1]:
        fields = line.removeprefix("file=").split("|") if line.startswith("file=") else []
        if (
            len(fields) != 3
            or fields[0] not in names
            or fields[0] in inventory
            or not fields[1].isdigit()
            or _SHA256.fullmatch(fields[2]) is None
        ):
            raise ExecutionMonitorError("REMOTE_HASH_PROTOCOL: invalid inventory record")
        size = _bounded_decimal(
            fields[1],
            label="REMOTE_HASH_PROTOCOL",
            maximum=MAX_FETCH_FILE_BYTES,
        )
        total += size
        if size > MAX_FETCH_FILE_BYTES or total > MAX_FETCH_TOTAL_BYTES:
            raise ExecutionMonitorError("FETCH_SIZE: remote results exceed size limits")
        inventory[fields[0]] = {"size": size, "sha256": fields[2]}
    if set(inventory) != set(names):
        raise ExecutionMonitorError("REMOTE_HASH_PROTOCOL: incomplete inventory")
    return inventory


def _bounded_decimal(value: str, *, label: str, maximum: int) -> int:
    if not value or len(value) > len(str(maximum)) or not value.isascii() or not value.isdigit():
        raise ExecutionMonitorError(f"{label}: invalid numeric field")
    number = int(value)
    if number > maximum:
        raise ExecutionMonitorError(f"{label}: numeric field exceeds its limit")
    return number


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _is_link_or_reparse(info: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(file_attributes & reparse_flag)


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: destination is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode) or _is_link_or_reparse(info):
        raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: destination was replaced")
    return _identity(info)


def _regular_file_digest(path: Path) -> tuple[int, str]:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: non-regular local result")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
                raise ExecutionMonitorError(
                    "FETCH_LOCAL_VERIFY: local result changed before reading"
                )
            if opened.st_size > MAX_FETCH_FILE_BYTES:
                raise ExecutionMonitorError("FETCH_SIZE: downloaded result exceeds its size limit")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            read_complete = os.fstat(handle.fileno())
        after = path.lstat()
    except ExecutionMonitorError:
        raise
    except OSError as exc:
        raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: cannot read local result safely") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(opened, field) != getattr(read_complete, field) for field in stable_fields):
        raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: local result changed while reading")
    if any(getattr(opened, field) != getattr(after, field) for field in stable_fields):
        raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: local result was replaced")
    return opened.st_size, digest.hexdigest()


def _verify_local(
    destination: Path,
    names: tuple[str, ...],
    remote: dict[str, dict[str, int | str]],
    directory_identity: tuple[int, int],
) -> dict[str, dict[str, int | str]]:
    if _directory_identity(destination) != directory_identity:
        raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: destination was replaced")
    entries = {item.name for item in destination.iterdir()}
    if entries != set(names):
        raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: unexpected or missing local files")
    verified: dict[str, dict[str, int | str]] = {}
    total = 0
    for name in names:
        path = destination / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != destination:
            raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: non-regular local result")
        size, digest = _regular_file_digest(path)
        total += size
        if size > MAX_FETCH_FILE_BYTES or total > MAX_FETCH_TOTAL_BYTES:
            raise ExecutionMonitorError("FETCH_SIZE: downloaded results exceed size limits")
        if size != remote[name]["size"] or digest != remote[name]["sha256"]:
            raise ExecutionMonitorError("FETCH_HASH_MISMATCH: local and remote results differ")
        verified[name] = {"size": size, "sha256": digest}
    if _directory_identity(destination) != directory_identity:
        raise ExecutionMonitorError("FETCH_LOCAL_VERIFY: destination changed during verification")
    return verified


def fetch_results(
    profile: ClusterProfile,
    record: RunRecord,
    destination: Path,
    filenames: Sequence[str],
    *,
    runner: Runner | None = None,
) -> dict:
    """Fetch exact allowlisted files into a newly-created local directory.

    Remote metadata is checked both before and after SFTP.  The operation
    fails if a file changes during transfer or if any local byte/hash differs.
    A failed fetch leaves its uniquely-created directory for manual audit.
    """

    preview = fetch_preview(profile, record, destination, filenames)
    names = tuple(preview["files"])
    ssh_argv = preview["ssh_inventory_argv"]
    ssh_stdin = preview["ssh_inventory_stdin"].encode("ascii")
    before = _parse_inventory(
        _execute(
            runner,
            ssh_argv,
            ssh_stdin,
            timeout=HASH_TIMEOUT_SECONDS,
            label="REMOTE_INVENTORY",
        ),
        names,
    )

    local = Path(preview["destination"])
    # The preview checked the parent and non-existence; mkdir remains the
    # atomic no-overwrite boundary in case another actor raced this process.
    parent_identity = _directory_identity(local.parent)
    try:
        local.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise ExecutionMonitorError("FETCH_DESTINATION: destination is no longer new") from exc
    if _directory_identity(local.parent) != parent_identity:
        raise ExecutionMonitorError("FETCH_DESTINATION: parent changed before creation")
    local_identity = _directory_identity(local)
    _execute(
        runner,
        preview["sftp_argv"],
        preview["sftp_batch"].encode("utf-8"),
        timeout=SFTP_TIMEOUT_SECONDS,
        label="SFTP_FETCH",
    )
    files = _verify_local(local, names, before, local_identity)
    after = _parse_inventory(
        _execute(
            runner,
            ssh_argv,
            ssh_stdin,
            timeout=HASH_TIMEOUT_SECONDS,
            label="REMOTE_REHASH",
        ),
        names,
    )
    if after != before:
        raise ExecutionMonitorError("REMOTE_RESULT_CHANGED: file changed during transfer")
    return {
        "schema_version": 1,
        "mode": "remote_fetch",
        "run_id": preview["run_id"],
        "node": preview["node"],
        "remote_directory": preview["remote_directory"],
        "destination": str(local),
        "files": files,
        "total_bytes": sum(int(item["size"]) for item in files.values()),
        "remote_hashes_verified": True,
        "local_hashes_verified": True,
        "potcar_fetched": False,
        "wavecar_fetched": False,
    }
