"""Fixed-argv remote MPI smoke and VASP start. No arbitrary shell from callers."""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    ExecutionRequest,
    SmokeTestEvidence,
)
from magnetic_vasp_agent.execution_runtime_models import SMOKE_CONFIRM
from magnetic_vasp_agent.transfer import _remote_script, _run_fixed
from magnetic_vasp_agent.transfer_models import TransferReport
from magnetic_vasp_agent.workflow import json_bytes
from magnetic_vasp_agent.workflow_models import safe_id

Runner = Callable[..., object]
SMOKE_RANKS = 2
SMOKE_MARKER = "MAGVASP_SMOKE_V2"
LAUNCH_MARKER = "MAGVASP_VASP"
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class RemoteExecutionError(ValueError):
    """Raised when a remote smoke/launch request is refused."""


def _setvars_path(profile: ClusterProfile) -> str | None:
    parts = profile.mpi_path.split("/")
    if "oneapi" in parts:
        return "/".join([*parts[: parts.index("oneapi") + 1], "setvars.sh"])
    return None


def _module_bootstrap(profile: ClusterProfile) -> str:
    setvars = _setvars_path(profile)
    lines = ["set +u\n"]
    if setvars:
        quoted = shlex.quote(setvars)
        lines.append(f"if test -r {quoted}; then . {quoted} --force >/dev/null; fi\n")
    lines.append(
        "if test -r /etc/profile.d/lmod.sh; then . /etc/profile.d/lmod.sh >/dev/null 2>&1 || true; "
        "elif test -r /etc/profile.d/modules.sh; then . /etc/profile.d/modules.sh >/dev/null 2>&1 || true; fi\n"
        f"if type module >/dev/null 2>&1; then module load {profile.vasp_module} >/dev/null 2>&1 || true; fi\n"
        "set -u\n"
    )
    return "".join(lines)


def _host_ok(observed: str, node: str) -> bool:
    return observed == node or observed.startswith(f"{node}.")


def smoke_preview(
    profile: ClusterProfile,
    node: str,
    *,
    run_id: str = "smoke",
    ranks: int = SMOKE_RANKS,
    cpu_ids: list[int] | None = None,
) -> dict:
    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    if node == profile.gateway_hostname:
        raise RemoteExecutionError("NODE_FORBIDDEN: gateway cannot run MPI")
    if node not in profile.allowed_nodes:
        raise RemoteExecutionError("NODE_FORBIDDEN: node is not in the configured cluster profile")
    selected = cpu_ids if cpu_ids is not None else list(range(ranks))
    if len(selected) != ranks or len(set(selected)) != ranks:
        raise RemoteExecutionError("CPU_BINDING_REQUIRED: one distinct CPU per smoke rank")
    pin = ",".join(str(item) for item in selected)
    rank_probe = (
        'cpu="$(/bin/ps -o psr= -p $$)"; '
        'cpu="${cpu//[[:space:]]/}"; '
        'printf "%s|%s\\n" "$(/bin/hostname -s)" "$cpu"'
    )
    mpi_argv = [
        profile.mpi_path,
        "-bootstrap",
        "fork",
        "-hosts",
        node,
        "-n",
        str(ranks),
        "/bin/bash",
        "-c",
        rank_probe,
    ]
    body = (
        _module_bootstrap(profile)
        + "export OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\n"
        + "export I_MPI_PIN=1\n"
        + f"export I_MPI_PIN_PROCESSOR_LIST={pin}\n"
        + "printf 'MAGVASP_SMOKE_V2\\n'\n"
        + shlex.join(mpi_argv)
        + " </dev/null\nprintf 'MAGVASP_SMOKE_END\\n'\n"
    )
    argv, stdin = _remote_script(profile, node, body)
    return {
        "mode": "mpi_smoke_preview",
        "run_id": run_id,
        "node": node,
        "ranks": ranks,
        "cpu_ids": selected,
        "ssh_argv": argv,
        "stdin_script": stdin.decode("utf-8"),
        "executes_vasp": False,
        "marker": SMOKE_MARKER,
    }


def _parse_smoke_ranks(
    stdout: bytes, node: str, ranks: int, cpu_ids: list[int]
) -> tuple[list[str], list[int]]:
    text = stdout.decode("utf-8", errors="strict")
    if text.count("MAGVASP_SMOKE_V2") != 1 or text.count("MAGVASP_SMOKE_END") != 1:
        raise RemoteExecutionError("SMOKE_PROTOCOL: missing smoke markers")
    block = text.split("MAGVASP_SMOKE_V2", 1)[1].split("MAGVASP_SMOKE_END", 1)[0]
    rows = [line.strip() for line in block.splitlines() if line.strip()]
    if len(rows) != ranks:
        raise RemoteExecutionError(
            f"SMOKE_PROTOCOL: expected {ranks} rank lines, got {len(rows)}"
        )
    hosts: list[str] = []
    observed_cpus: list[int] = []
    for row in rows:
        fields = row.split("|")
        if len(fields) != 2 or not fields[1].isascii() or not fields[1].isdigit():
            raise RemoteExecutionError("SMOKE_PROTOCOL: malformed rank observation")
        host, cpu_text = fields
        if HOST_RE.fullmatch(host) is None or not _host_ok(host, node):
            raise RemoteExecutionError("SMOKE_CROSS_NODE: smoke left the approved single host")
        hosts.append(host)
        observed_cpus.append(int(cpu_text))
    if len(set(observed_cpus)) != ranks or set(observed_cpus) != set(cpu_ids):
        raise RemoteExecutionError("SMOKE_CPU_BINDING: ranks did not use the approved CPUs")
    return hosts, observed_cpus


def run_mpi_smoke(
    profile: ClusterProfile,
    node: str,
    *,
    confirm: str,
    run_id: str = "smoke",
    ranks: int = SMOKE_RANKS,
    cpu_ids: list[int] | None = None,
    runner: Callable[..., object] | None = None,
) -> dict:
    if confirm != SMOKE_CONFIRM:
        raise RemoteExecutionError("smoke requires the exact phrase I_CONFIRM_MPI_SMOKE")
    if not profile.execution_enabled:
        raise RemoteExecutionError("REMOTE_EXECUTION_DISABLED")
    preview = smoke_preview(
        profile, node, run_id=run_id, ranks=ranks, cpu_ids=cpu_ids
    )
    execute = runner or _run_fixed
    started_at = datetime.now(UTC)
    result = execute(preview["ssh_argv"], preview["stdin_script"].encode("utf-8"), timeout=90)
    finished_at = datetime.now(UTC)
    if finished_at <= started_at:
        finished_at = started_at + timedelta(microseconds=1)
    stdout = getattr(result, "stdout", b"") or b""
    stderr = getattr(result, "stderr", b"") or b""
    code = int(getattr(result, "returncode", 1))
    hosts: list[str] = []
    rank_cpu_ids: list[int] = []
    try:
        if code != 0:
            raise RemoteExecutionError(f"SMOKE_FAILED: exit {code}")
        hosts, rank_cpu_ids = _parse_smoke_ranks(
            stdout, node, ranks, preview["cpu_ids"]
        )
    except RemoteExecutionError as exc:
        raise RemoteExecutionError(str(exc).split(":", 1)[0]) from exc
    evidence = SmokeTestEvidence(
        run_id=run_id,
        node=node,
        mpi_ranks=ranks,
        cpu_ids=preview["cpu_ids"],
        hostname=hosts[0],
        started_at=started_at,
        finished_at=finished_at,
        exit_code=code,
        timed_out=False,
        rank_cpu_ids=rank_cpu_ids,
        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        output_bytes=len(stdout) + len(stderr),
        passed=True,
    )
    return evidence.model_dump(mode="json")


def _step_directory(remote_root_leaf: str, step_id: str) -> str:
    safe_id(step_id)
    return f"{remote_root_leaf}/{step_id}"


def launch_preview(
    profile: ClusterProfile,
    request: ExecutionRequest,
    transfer: TransferReport,
    *,
    executable: str,
    cpu_ids: list[int],
    artifact_sha256: str | None = None,
    artifact_size: int | None = None,
) -> dict:
    if request.node == profile.gateway_hostname:
        raise RemoteExecutionError("NODE_FORBIDDEN: gateway cannot run VASP")
    if request.node not in profile.allowed_nodes:
        raise RemoteExecutionError("NODE_FORBIDDEN: node is not in the configured cluster profile")
    if request.node != transfer.node:
        raise RemoteExecutionError("TRANSFER_NODE_MISMATCH")
    if not transfer.transferred or not transfer.remote_hashes_verified:
        raise RemoteExecutionError("TRANSFER_NOT_VERIFIED")
    if executable not in profile.vasp_paths:
        raise RemoteExecutionError("VASP_UNAVAILABLE")
    if len(cpu_ids) != request.mpi_ranks or len(set(cpu_ids)) != request.mpi_ranks:
        raise RemoteExecutionError("CPU_BINDING_REQUIRED")
    if (artifact_sha256 is None) != (artifact_size is None):
        raise RemoteExecutionError("ARTIFACT_EVIDENCE_INCOMPLETE")
    if artifact_sha256 is not None and (
        re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
        or artifact_size is None
        or not 1 <= artifact_size <= 2**31
    ):
        raise RemoteExecutionError("ARTIFACT_EVIDENCE_INVALID")
    workdir = _step_directory(transfer.remote_directory, request.step_id)
    quoted = shlex.quote(workdir)
    pin = ",".join(str(item) for item in cpu_ids)
    vasp = profile.vasp_paths[executable]
    mpi_argv = [
        profile.mpi_path,
        "-bootstrap",
        "fork",
        "-hosts",
        request.node,
        "-n",
        str(request.mpi_ranks),
        vasp,
    ]
    marker = f"{LAUNCH_MARKER} {request.run_id}"
    artifact_guard = ""
    if artifact_sha256 is not None:
        artifact_guard = (
            f"test -f {quoted}/CHGCAR && test ! -L {quoted}/CHGCAR || exit 95\n"
            f"test \"$(stat -c %s -- {quoted}/CHGCAR)\" = {artifact_size} || exit 96\n"
            f"test \"$(sha256sum -- {quoted}/CHGCAR | cut -d' ' -f1)\" = {artifact_sha256} || exit 97\n"
        )
    body = (
        _module_bootstrap(profile)
        + f"test -d {quoted} || exit 93\n"
        + f"test -f {quoted}/INCAR || exit 93\n"
        + f"test -f {quoted}/POSCAR || exit 93\n"
        + f"test -f {quoted}/POTCAR || exit 93\n"
        + f"test -f {quoted}/KPOINTS || exit 93\n"
        + artifact_guard
        + f"test ! -e {quoted}/magvasp.launch.pid || exit 94\n"
        + f"cd -- {quoted}\n"
        + "export OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\n"
        + "export I_MPI_PIN=1\n"
        + f"export I_MPI_PIN_PROCESSOR_LIST={pin}\n"
        + f"export MAGVASP_RUN_MARKER={marker.replace(' ', '_')}\n"
        + "nohup "
        + " ".join(mpi_argv)
        + " </dev/null > magvasp.stdout.log 2> magvasp.stderr.log &\n"
        + "printf '%s\\n' $! > magvasp.launch.pid\n"
        + "printf 'MAGVASP_LAUNCH_V1\\n'\n"
        + "printf 'pid=%s\\n' \"$(cat magvasp.launch.pid)\"\n"
        + "printf 'cwd=%s\\n' \"$(pwd)\"\n"
        + "printf 'host=%s\\n' \"$(hostname -s)\"\n"
        + "printf 'MAGVASP_LAUNCH_END\\n'\n"
    )
    argv, stdin = _remote_script(profile, request.node, body)
    return {
        "mode": "remote_vasp_preview",
        "node": request.node,
        "workdir": workdir,
        "executable": executable,
        "mpi_argv": mpi_argv,
        "cpu_ids": cpu_ids,
        "ssh_argv": argv,
        "stdin_script": stdin.decode("utf-8"),
        "marker": marker,
    }


def _parse_launch(stdout: bytes, node: str) -> tuple[int, str]:
    text = stdout.decode("utf-8", errors="strict")
    if text.count("MAGVASP_LAUNCH_V1") != 1 or text.count("MAGVASP_LAUNCH_END") != 1:
        raise RemoteExecutionError("LAUNCH_PROTOCOL: missing launch markers")
    block = text.split("MAGVASP_LAUNCH_V1", 1)[1].split("MAGVASP_LAUNCH_END", 1)[0]
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    pid_text = fields.get("pid", "")
    host = fields.get("host", "")
    if not pid_text.isdigit() or int(pid_text) <= 0:
        raise RemoteExecutionError("LAUNCH_PROTOCOL: missing remote pid")
    if not _host_ok(host, node):
        raise RemoteExecutionError("LAUNCH_HOST_MISMATCH")
    return int(pid_text), fields.get("cwd", "")


def start_remote_vasp(
    profile: ClusterProfile,
    request: ExecutionRequest,
    transfer: TransferReport,
    *,
    executable: str,
    cpu_ids: list[int],
    artifact_sha256: str | None = None,
    artifact_size: int | None = None,
    runner: Callable[..., object] | None = None,
) -> dict:
    if not profile.execution_enabled:
        raise RemoteExecutionError("REMOTE_EXECUTION_DISABLED")
    preview = launch_preview(
        profile,
        request,
        transfer,
        executable=executable,
        cpu_ids=cpu_ids,
        artifact_sha256=artifact_sha256,
        artifact_size=artifact_size,
    )
    execute = runner or _run_fixed
    result = execute(preview["ssh_argv"], preview["stdin_script"].encode("utf-8"), timeout=60)
    code = int(getattr(result, "returncode", 1))
    stdout = getattr(result, "stdout", b"") or b""
    if code != 0:
        raise RemoteExecutionError(f"LAUNCH_FAILED: exit {code}")
    pid, cwd = _parse_launch(stdout, request.node)
    return {
        "schema_version": 1,
        "mode": "remote_vasp_start",
        "node": request.node,
        "workdir": preview["workdir"],
        "cwd": cwd,
        "remote_pid": pid,
        "executable": executable,
        "mpi_argv": preview["mpi_argv"],
        "cpu_ids": cpu_ids,
        "marker": preview["marker"],
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
    }


def write_json_report(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(json_bytes(payload))
