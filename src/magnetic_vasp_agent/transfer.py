"""Fail-closed SFTP transfer of immutable input packages.

This module deliberately has no generic remote-command interface. The only SSH
commands are fixed identity/directory guards and fixed SHA256 verification.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from magnetic_vasp_agent.execution_models import ALLOWED_NODES, ClusterProfile
from magnetic_vasp_agent.execution_probe import SSH_OPTIONS
from magnetic_vasp_agent.transfer_models import (
    TransferFile,
    TransferInventory,
    TransferReport,
    TransferRequest,
)
from magnetic_vasp_agent.workflow import INPUT_NAMES, inspect_bundle, json_bytes
from magnetic_vasp_agent.workflow_models import safe_id

MAX_FILE_BYTES = 2**31
MAX_TOTAL_BYTES = 2**32
MAX_COMMAND_BYTES = 256_000
MAX_OUTPUT_BYTES = 2_000_000
TRANSFER_FILENAMES = frozenset((*INPUT_NAMES, "manifest.json", "workflow.manifest.json"))
SFTP_OPTIONS = [
    "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
    "-o", "ControlMaster=no", "-o", "ControlPath=none",
    "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=2", "-o", "PermitLocalCommand=no",
]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sftp_quote(value: str) -> str:
    """Quote one SFTP batch token without turning it into a shell command."""
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("SFTP path contains an invalid control character")
    return shlex.quote(value.replace("\\", "/"))


def _remote_leaf(profile: ClusterProfile, workflow_id: str, run_id: str) -> str:
    if profile.remote_root is None:
        raise ValueError("REMOTE_ROOT_REQUIRED: configure a dedicated existing remote directory")
    safe_id(workflow_id)
    safe_id(run_id)
    # One new leaf makes the no-overwrite boundary atomic at directory creation.
    leaf = f"{workflow_id}--{run_id}"
    remote = f"{profile.remote_root}/{leaf}"
    if len(remote) > 512:
        raise ValueError("remote destination path is too long")
    return remote


def _inventory(bundle: Path) -> TransferInventory:
    bundle = Path(bundle)
    if bundle.is_symlink():
        raise ValueError("TRANSFER_SCOPE: workflow bundle must not be a symlink")
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise ValueError("TRANSFER_SCOPE: workflow bundle must be a directory")
    inspect_bundle(bundle)
    top_path = bundle / "workflow.manifest.json"
    if top_path.is_symlink():
        raise ValueError("TRANSFER_SCOPE: workflow manifest must not be a symlink")
    top = json.loads(top_path.read_text(encoding="utf-8"))
    workflow_id = safe_id(top["workflow_id"])
    order = top.get("order")
    steps = top.get("steps")
    if not isinstance(order, list) or not isinstance(steps, dict) or set(order) != set(steps):
        raise ValueError("TRANSFER_MANIFEST: workflow order and steps are malformed")
    expected_top = {"workflow.manifest.json", *steps}
    actual_top = {item.name for item in bundle.iterdir()}
    if actual_top != expected_top:
        raise ValueError("TRANSFER_SCOPE: bundle contains unexpected top-level files")
    files: list[TransferFile] = []
    total = 0
    for name in [*order, "workflow.manifest.json"]:
        safe_id(name) if name != "workflow.manifest.json" else None
        folder = bundle if name == "workflow.manifest.json" else bundle / name
        if folder.is_symlink() or (folder.resolve().parent != bundle if name != "workflow.manifest.json" else False):
            raise ValueError("TRANSFER_SCOPE: symlinks are not transferable")
        names = ["workflow.manifest.json"] if name == "workflow.manifest.json" else [*INPUT_NAMES, "manifest.json"]
        if name == "workflow.manifest.json":
            paths = [folder / "workflow.manifest.json"]
        else:
            if not folder.is_dir():
                raise ValueError("TRANSFER_SCOPE: step entry is not a directory")
            if {item.name for item in folder.iterdir()} != set(names):
                raise ValueError(f"TRANSFER_SCOPE: {name} contains unexpected files")
            paths = [folder / filename for filename in names]
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.resolve().parent not in {bundle, folder.resolve()}:
                raise ValueError("TRANSFER_SCOPE: only regular files inside the bundle may be uploaded")
            size = path.stat().st_size
            total += size
            if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("TRANSFER_SIZE: package exceeds transfer size limits")
            relative = path.relative_to(bundle).as_posix()
            if relative.split("/")[-1] not in TRANSFER_FILENAMES:
                raise ValueError("TRANSFER_SCOPE: unapproved filename")
            files.append(TransferFile(relative_path=relative, size=size, sha256=_sha256_file(path)))
    return TransferInventory(workflow_id=workflow_id, files=files, total_bytes=total)


def _nested_ssh_argv(profile: ClusterProfile, node: str, remote_command: str) -> list[str]:
    if node not in ALLOWED_NODES:
        raise ValueError("NODE_FORBIDDEN: only the nine group compute nodes are allowed")
    if len(remote_command.encode("utf-8")) > MAX_COMMAND_BYTES:
        raise ValueError("remote command is too large")
    gateway_script = (
        'test "$(hostname -s)" = mu01 || exit 91\n'
        'test "$(id -un)" = zyy || exit 92\n'
        "exec ssh " + shlex.join([
            *SSH_OPTIONS, "-p", str(profile.port), f"zyy@{node}",
            "/bin/bash -s",
        ])
    )
    return [
        "ssh", *SSH_OPTIONS, "-p", str(profile.port),
        f"{profile.user}@{profile.gateway}",
        "/bin/bash -c " + shlex.quote(gateway_script),
    ]


def _remote_script(profile: ClusterProfile, node: str, body: str) -> tuple[list[str], bytes]:
    argv = _nested_ssh_argv(profile, node, "/bin/bash -s")
    script = (
        "set -eu\nexport LC_ALL=C\n"
        f'test "$(hostname -s)" = {node} || exit 91\n'
        'test "$(id -un)" = zyy || exit 92\n'
        f"{body}\n"
    ).encode()
    return argv, script


def _sftp_argv(profile: ClusterProfile, node: str) -> list[str]:
    if node not in ALLOWED_NODES:
        raise ValueError("NODE_FORBIDDEN: only the nine group compute nodes are allowed")
    return [
        "sftp", *SFTP_OPTIONS, "-o", f"ProxyJump={profile.user}@{profile.gateway}:{profile.port}",
        "-P", str(profile.port), f"{profile.user}@{node}",
    ]


def _remote_guard_script(remote: str) -> str:
    quoted = shlex.quote(remote)
    parent = shlex.quote(str(PurePosixPath(remote).parent))
    return (
        f"test -d {parent}\n"
        f"test ! -L {parent}\n"
        f"test ! -e {quoted}\n"
        f"test ! -L {quoted}\n"
        f"mkdir -- {quoted}\n"
    )


def _remote_hash_script(remote: str, files: list[TransferFile]) -> str:
    lines = ["printf 'MAGVASP_TRANSFER_V1\\n'"]
    for item in files:
        lines.append(f"sha256sum -- {shlex.quote(remote + '/' + item.relative_path)}")
    return "\n".join(lines)


def _batch(bundle: Path, remote: str, files: list[TransferFile]) -> str:
    lines = []
    made = set()
    for item in files:
        remote_file = remote + "/" + item.relative_path
        parent = remote_file.rsplit("/", 1)[0]
        if parent != remote and parent not in made:
            # The leaf itself was created by the atomic remote guard.
            lines.append(f"mkdir {_sftp_quote(parent)}")
            made.add(parent)
        lines.append(
            f"put {_sftp_quote(str((bundle / item.relative_path).resolve()))} {_sftp_quote(remote_file)}"
        )
    lines.append("bye")
    return "\n".join(lines) + "\n"


def build_transfer_plan(bundle: Path, profile: ClusterProfile, request: TransferRequest) -> dict:
    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    request = TransferRequest.model_validate_json(request.model_dump_json())
    inventory = _inventory(bundle)
    remote = _remote_leaf(profile, inventory.workflow_id, request.run_id)
    guard_argv, guard_stdin = _remote_script(
        profile, request.node, _remote_guard_script(remote)
    )
    hash_argv, hash_stdin = _remote_script(
        profile, request.node, _remote_hash_script(remote, inventory.files)
    )
    batch = _batch(bundle.resolve(), remote, inventory.files)
    return {
        "schema_version": 1,
        "mode": "transfer_plan",
        "workflow_id": inventory.workflow_id,
        "run_id": request.run_id,
        "node": request.node,
        "local_bundle": str(Path(bundle).resolve()),
        "remote_directory": remote,
        "files": [item.model_dump() for item in inventory.files],
        "total_bytes": inventory.total_bytes,
        "ready_to_transfer": True,
        "transferred": False,
        "remote_hashes_verified": False,
        "remote_directory_created": False,
        "execution_enabled": False,
        "ssh_guard_argv": guard_argv,
        "ssh_guard_stdin": guard_stdin.decode("utf-8"),
        "sftp_argv": _sftp_argv(profile, request.node),
        "sftp_batch": batch,
        "ssh_hash_argv": hash_argv,
        "ssh_hash_stdin": hash_stdin.decode("utf-8"),
        "blockers": [],
        "limitations": [
            "The remote_root must already exist and be group-approved; this tool never creates it",
            "A failed upload leaves its unique remote directory for manual inspection; it is never deleted automatically",
            "Transfer verification proves bytes at upload time, not VASP runtime, convergence or artifact compatibility",
            "This operation transfers only the four VASP inputs and manifests; CHGCAR/WAVECAR are never invented or uploaded",
        ],
    }


def _run_fixed(argv: list[str], stdin: bytes, *, timeout: int) -> subprocess.CompletedProcess:
    executable = shutil.which(argv[0])
    if executable is None:
        raise ValueError(f"{argv[0].upper()}_UNAVAILABLE: install native OpenSSH first")
    actual = [executable, *argv[1:]]
    try:
        result = subprocess.run(
            actual, input=stdin, capture_output=True, timeout=timeout,
            check=False, shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{argv[0].upper()}_TIMEOUT: transfer operation timed out") from exc
    if len(result.stdout) + len(result.stderr) > MAX_OUTPUT_BYTES:
        raise ValueError(f"{argv[0].upper()}_OUTPUT_TOO_LARGE: refusing unbounded remote output")
    return result


def _check_success(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode:
        raise ValueError(f"{label}_FAILED: exit {result.returncode}; inspect the remote directory manually")


def _parse_hashes(output: bytes, files: list[TransferFile]) -> dict[str, str]:
    text = output.decode("utf-8", errors="strict")
    lines = text.splitlines()
    if lines.count("MAGVASP_TRANSFER_V1") != 1:
        raise ValueError("REMOTE_HASH_PROTOCOL: missing or duplicated verification header")
    found: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    for line in lines:
        if line == "MAGVASP_TRANSFER_V1":
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError("REMOTE_HASH_PROTOCOL: unexpected verification output")
        digest_value, path = match.groups()
        if path in found:
            raise ValueError("REMOTE_HASH_PROTOCOL: duplicate remote path")
        found[path] = digest_value
    expected = {item.relative_path: item.sha256 for item in files}
    # sha256sum reports the absolute path; map it only after exact suffix checking.
    normalized = {}
    for path, value in found.items():
        matches = [item.relative_path for item in files if path.endswith("/" + item.relative_path)]
        if len(matches) != 1:
            raise ValueError("REMOTE_HASH_PROTOCOL: unrecognized remote path")
        normalized[matches[0]] = value
    if normalized != expected:
        raise ValueError("REMOTE_HASH_MISMATCH: uploaded bytes differ from the local inventory")
    return normalized


def transfer_bundle(bundle: Path, profile: ClusterProfile, request: TransferRequest) -> dict:
    plan = build_transfer_plan(bundle, profile, request)
    files = [TransferFile.model_validate(item) for item in plan["files"]]
    guard = _run_fixed(
        plan["ssh_guard_argv"], plan["ssh_guard_stdin"].encode("utf-8"), timeout=30
    )
    _check_success(guard, "REMOTE_DIRECTORY_GUARD")
    upload = _run_fixed(
        plan["sftp_argv"], plan["sftp_batch"].encode("utf-8"), timeout=120
    )
    _check_success(upload, "SFTP_UPLOAD")
    verification = _run_fixed(
        plan["ssh_hash_argv"], plan["ssh_hash_stdin"].encode("utf-8"), timeout=60
    )
    _check_success(verification, "REMOTE_HASH")
    _parse_hashes(verification.stdout, files)
    return {
        **plan,
        "mode": "transfer",
        "transferred": True,
        "remote_hashes_verified": True,
        "remote_directory_created": True,
        "ready_to_transfer": False,
    }


def transfer_report_model(report: dict) -> TransferReport:
    return TransferReport.model_validate({
        key: report[key]
        for key in (
            "schema_version", "mode", "workflow_id", "run_id", "node", "local_bundle",
            "remote_directory", "files", "ready_to_transfer", "transferred",
            "remote_hashes_verified", "remote_directory_created", "execution_enabled",
            "blockers", "limitations",
        )
    })


def write_transfer_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(json_bytes(report))
