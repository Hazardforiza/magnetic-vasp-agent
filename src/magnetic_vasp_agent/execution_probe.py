"""Fixed read-only probe. No upload, MPI launch, process signals or arbitrary commands."""

from __future__ import annotations

import base64
import json
import re
import shlex
import shutil
import subprocess
from datetime import UTC, datetime

from magnetic_vasp_agent.execution_models import (
    ALLOWED_NODES,
    VASP_PATHS,
    WANNIER_MODULE,
    Blocker,
    ClusterProfile,
    Cpu,
    ExecutionRequest,
    ProbeSnapshot,
    ProcessObservation,
    ResourceAssessment,
    WannierProbeSnapshot,
)

SECTIONS = ("identity", "time", "cpus", "affinity", "processes", "load", "memory", "runtime")
MAX_PROBE_BYTES = 2_000_000
WANNIER_SECTIONS = ("identity", "runtime")
MAX_WANNIER_PROBE_BYTES = 64_000
# Shared hardening. Authentication is layered on by the batch / interactive presets.
_SSH_HARDENING = [
    "-T",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ForwardAgent=no",
    "-o", "ClearAllForwardings=yes",
    "-o", "PermitLocalCommand=no",
    "-o", "ConnectTimeout=8",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=2",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
]
# Unattended path: keys only, never a password prompt (LLM / CI).
SSH_OPTIONS = [
    *_SSH_HARDENING,
    "-o", "BatchMode=yes",
]
# Human path: same as MobaXterm — type the password in the terminal; never stored.
# No -T: Windows OpenSSH reads the password from the console, not from a pipe.
SSH_OPTIONS_INTERACTIVE = [
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ForwardAgent=no",
    "-o", "ClearAllForwardings=yes",
    "-o", "PermitLocalCommand=no",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=2",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "BatchMode=no",
    "-o", "PreferredAuthentications=password,keyboard-interactive",
    "-o", "PubkeyAuthentication=no",
    "-o", "NumberOfPasswordPrompts=1",
]


def _probe_script(profile: ClusterProfile, node: str) -> str:
    script = f"""set -eu
export LC_ALL=C
MAGVASP_PROBE_STAGE=identity
probe_error() {{ printf 'MAGVASP_PROBE_ERROR:%s\\n' \"$MAGVASP_PROBE_STAGE\" >&2; }}
trap probe_error ERR
# Gateway identity is checked by the outer hop: {profile.gateway_hostname}
test "$(hostname -s)" = {node} || exit 91
test "$(id -un)" = {shlex.quote(profile.user)} || exit 92
# Do not source /etc/profile here: this cluster's login startup is interactive-only.
# Module initialization is handled by the explicit fixed checks below.
MAGVASP_PROBE_STAGE=identity_output
printf 'MAGVASP_PROBE_V1\\n@@identity\\n'
hostname
id -un
printf '@@time\\n'
MAGVASP_PROBE_STAGE=time
date -u +%Y-%m-%dT%H:%M:%SZ
printf '@@cpus\\n'
MAGVASP_PROBE_STAGE=cpus
lscpu --json --extended=CPU,CORE,SOCKET,ONLINE
printf '@@affinity\\n'
MAGVASP_PROBE_STAGE=affinity
awk '/^Cpus_allowed_list:/ {{print $2}}' /proc/self/status
printf '@@processes\\n'
MAGVASP_PROBE_STAGE=processes
ps -eo pid=,user=,psr=,pcpu=,comm=
printf '@@load\\n'
MAGVASP_PROBE_STAGE=load
cat /proc/loadavg
printf '@@memory\\n'
MAGVASP_PROBE_STAGE=memory
awk '/^MemAvailable:/ {{print $2}}' /proc/meminfo
printf '@@runtime\\n'
MAGVASP_PROBE_STAGE=runtime
if ! type module >/dev/null 2>&1; then
    if test {shlex.quote(profile.lmod_path or '')} != "" && test -x {shlex.quote(profile.lmod_path or '/nonexistent')}; then
        export LMOD_CMD={shlex.quote(profile.lmod_path or '')}
        export LMOD_SETTARG_CMD=:
        if test {shlex.quote(profile.modulepath or '')} != ""; then export MODULEPATH={shlex.quote(profile.modulepath or '')}; fi
        module() {{
            eval "$(\"$LMOD_CMD\" shell \"$@\")"
        }}
    fi
fi
if ! type module >/dev/null 2>&1; then
    if test -r /etc/profile.d/lmod.sh; then
        source /etc/profile.d/lmod.sh
    elif test -r /etc/profile.d/modules.sh; then
        source /etc/profile.d/modules.sh
    fi
fi
MODULE_OK=false
if type module >/dev/null 2>&1 && module load {profile.vasp_module} >/dev/null 2>&1; then
    MODULE_OK=true
fi
# Some cluster images expose the VASP installation globally while Lmod is
# only available in interactive startup files. Treat the fully verified
# configured runtime as equivalent for the read-only probe.
if ! $MODULE_OK && test -x {shlex.quote(profile.mpi_path or '/nonexistent')}; then
    MODULE_OK=true
fi
if $MODULE_OK; then
    printf 'module_loaded=yes\\n'
else
    printf 'module_loaded=no\\n'
fi
MPI_DETECTED="$(command -v mpirun || true)"
if test -z "$MPI_DETECTED" && test -x {shlex.quote(profile.mpi_path or '/nonexistent')}; then
    MPI_DETECTED={shlex.quote(profile.mpi_path or '')}
fi
printf 'mpi_path=%s\\n' "$MPI_DETECTED"
"""
    for name, path in profile.vasp_paths.items():
        script += f"if test -x {path}; then printf '{name}=yes\\n'; else printf '{name}=no\\n'; fi\n"
    return script + "MAGVASP_PROBE_STAGE=complete\nprintf '@@end\\n'\n"


def _wannier_probe_script(profile: ClusterProfile, node: str) -> str:
    """Build the fixed, read-only Wannier capability probe.

    The module load is the only environment mutation and exists only in the
    child shell. ``wannier90.x -v`` is a banner query; no seed file, VASP
    executable, MPI launcher, or remote file is supplied to the process.
    """

    module = shlex.quote(profile.wannier_module)
    lmod_path = shlex.quote(profile.lmod_path or "")
    modulepath = shlex.quote(profile.modulepath or "")
    return f"""set -eu
export LC_ALL=C
MAGVASP_PROBE_STAGE=identity
probe_error() {{ printf 'MAGVASP_WANNIER_PROBE_ERROR:%s\\n' "$MAGVASP_PROBE_STAGE" >&2; }}
trap probe_error ERR
test "$(hostname -s)" = {shlex.quote(node)} || exit 91
test "$(id -un)" = {shlex.quote(profile.user)} || exit 92
printf 'MAGVASP_WANNIER_PROBE_V1\\n@@identity\\n'
hostname
id -un
printf '@@runtime\\n'
MAGVASP_PROBE_STAGE=runtime
# Initialise Lmod only through configured, fixed paths. No shell startup file
# or user-provided command is sourced by this probe.
if ! type module >/dev/null 2>&1; then
    if test {lmod_path} != "" && test -x {lmod_path}; then
        export LMOD_CMD={lmod_path}
        export LMOD_SETTARG_CMD=:
        if test {modulepath} != ""; then export MODULEPATH={modulepath}; fi
        module() {{ eval "$LMOD_CMD" shell "$@"; }}
    fi
fi
if ! type module >/dev/null 2>&1; then
    if test -r /etc/profile.d/lmod.sh; then
        source /etc/profile.d/lmod.sh
    elif test -r /etc/profile.d/modules.sh; then
        source /etc/profile.d/modules.sh
    fi
fi
MODULE_OK=false
if type module >/dev/null 2>&1 && module load {module} >/dev/null 2>&1; then
    MODULE_OK=true
fi
WANNIER90_PATH=""
WANNIER90_VERSION=""
VASP_NCL_PATH=""
if $MODULE_OK; then
    WANNIER90_PATH="$(command -v wannier90.x || true)"
    VASP_NCL_PATH="$(command -v vasp_ncl || true)"
    if test -n "$WANNIER90_PATH"; then
        # This fixed banner query must not receive a seed name or input file.
        WANNIER90_VERSION="$("$WANNIER90_PATH" -v 2>&1 | head -c 256 | tr '\\n' ' ' | tr -d '\\r' || true)"
    fi
fi
printf 'module_name=%s\\n' {module}
if $MODULE_OK; then printf 'module_loaded=yes\\n'; else printf 'module_loaded=no\\n'; fi
printf 'wannier90_path=%s\\n' "$WANNIER90_PATH"
printf 'wannier90_version=%s\\n' "$WANNIER90_VERSION"
printf 'vasp_ncl_path=%s\\n' "$VASP_NCL_PATH"
MAGVASP_PROBE_STAGE=complete
printf '@@end\\n'
"""


def _gateway_wrapper(profile: ClusterProfile, node: str, script: str) -> str:
    """Embed the compute-node script so stdin stays free for a password prompt.

    Intra-cluster ssh (mu01 → cuXX) stays BatchMode: that hop is host-key based
    after you are already on the gateway, and must not steal the password TTY.
    """

    token = base64.b64encode(script.encode("utf-8")).decode("ascii")
    inner = [
        "ssh", *SSH_OPTIONS, "-p", str(profile.node_port), f"{profile.user}@{node}",
        f"echo {token} | base64 -d | /bin/bash",
    ]
    return (
        f'test "$(hostname -s)" = {shlex.quote(profile.gateway_hostname)} || exit 91\n'
        f'test "$(id -un)" = {shlex.quote(profile.user)} || exit 92\n'
        "exec " + shlex.join(inner)
    )


def wannier_probe_preview(
    profile: ClusterProfile, node: str, *, interactive: bool = False
) -> dict:
    """Return a non-executing preview of the fixed Wannier capability probe."""

    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    if node not in profile.allowed_nodes:
        raise ValueError("NODE_FORBIDDEN: node is not in the configured cluster profile")
    script = _wannier_probe_script(profile, node)
    outer = SSH_OPTIONS_INTERACTIVE if interactive else SSH_OPTIONS
    argv = [
        "ssh", *outer, "-p", str(profile.port),
        f"{profile.user}@{profile.gateway}",
        "/bin/bash -c " + shlex.quote(_gateway_wrapper(profile, node, script)),
    ]
    return {
        "mode": "readonly_wannier_probe_preview",
        "node": node,
        "module": profile.wannier_module,
        "auth_mode": "interactive_password" if interactive else "batch_publickey",
        "ssh_argv": argv,
        "stdin_script": script,
        "stdin_used": False,
        "timeout_seconds": 30,
        "executes_mpi": False,
        "executes_vasp": False,
        "executes_wannier_calculation": False,
        "writes_remote_files": False,
        "stores_password": False,
    }


_WANNIER_VERSION_RE = re.compile(
    r"(?i)\b(?:wannier90\s*[:.-]?\s*)?(?:version\s*)?"
    r"v?([0-9]+(?:\.[0-9]+){1,3}(?:[-+._][A-Za-z0-9.-]+)?)\b"
)


def _normalise_wannier_version(value: str) -> str | None:
    match = _WANNIER_VERSION_RE.search(value.strip())
    return match.group(1) if match else None


def _parse_wannier_path(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if (
        not value.startswith("/")
        or len(value) > 512
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(char in value for char in "'\";|&$`<>\\")
    ):
        raise ValueError("invalid executable path in Wannier probe")
    return value


def parse_wannier_probe(
    output: str,
    node: str,
    *,
    allowed_nodes: tuple[str, ...] = ALLOWED_NODES,
    expected_module: str = WANNIER_MODULE,
) -> WannierProbeSnapshot:
    """Parse only the versioned, bounded Wannier probe protocol."""

    if node not in allowed_nodes or len(output.encode("utf-8")) > MAX_WANNIER_PROBE_BYTES:
        raise ValueError("invalid Wannier probe target or oversized output")
    lines = output.splitlines()
    if lines.count("MAGVASP_WANNIER_PROBE_V1") != 1:
        raise ValueError("missing or duplicated Wannier probe header")
    lines = lines[lines.index("MAGVASP_WANNIER_PROBE_V1") + 1:]
    sections: dict[str, list[str]] = {}
    current = None
    markers: list[str] = []
    for line in lines:
        if line.startswith("@@"):
            current = line[2:]
            if current in sections:
                raise ValueError("duplicate Wannier probe section")
            sections[current] = []
            markers.append(current)
        elif current is None:
            raise ValueError("unexpected data before Wannier probe section")
        else:
            sections[current].append(line)
    if markers != [*WANNIER_SECTIONS, "end"] or sections["end"]:
        raise ValueError("incomplete or malformed Wannier probe sections")
    try:
        identity = sections["identity"]
        if len(identity) != 2:
            raise ValueError("invalid Wannier probe identity section")
        runtime = dict(line.split("=", 1) for line in sections["runtime"])
        required = {
            "module_name", "module_loaded", "wannier90_path",
            "wannier90_version", "vasp_ncl_path",
        }
        if len(sections["runtime"]) != len(required) or set(runtime) != required:
            raise ValueError("unexpected Wannier runtime probe fields")
        if runtime["module_loaded"] not in {"yes", "no"}:
            raise ValueError("invalid Wannier module marker")
        module_name = runtime["module_name"]
        if module_name != expected_module:
            raise ValueError("Wannier probe module does not match the configured profile")
        raw_version = runtime["wannier90_version"].strip()
        version = _normalise_wannier_version(raw_version) if raw_version else None
        if raw_version and version is None:
            raise ValueError("Wannier version banner has no recognised version")
        return WannierProbeSnapshot(
            source="ssh_wannier_probe",
            observed_at=datetime.now(UTC),
            requested_node=node,
            hostname=identity[0],
            username=identity[1],
            module_name=module_name,
            module_loaded=runtime["module_loaded"] == "yes",
            wannier90_path=_parse_wannier_path(runtime["wannier90_path"]),
            wannier90_version=version,
            vasp_ncl_path=_parse_wannier_path(runtime["vasp_ncl_path"]),
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed Wannier probe payload") from exc


def collect_wannier_probe(
    profile: ClusterProfile, node: str, *, interactive: bool = False
) -> WannierProbeSnapshot:
    """Run the fixed read-only Wannier probe through the configured gateway."""

    preview = wannier_probe_preview(profile, node, interactive=interactive)
    result = _run_ssh(
        [_ssh_executable(), *preview["ssh_argv"][1:]],
        timeout=preview["timeout_seconds"],
        interactive=interactive,
    )
    if result.returncode:
        stderr = (result.stderr or b"").decode("utf-8", errors="ignore")
        stage = next(
            (
                line.split(":", 1)[1]
                for line in stderr.splitlines()
                if line.startswith("MAGVASP_WANNIER_PROBE_ERROR:")
                and line.count(":") == 1
            ),
            None,
        )
        suffix = f" at fixed stage {stage}" if stage else ""
        raise ValueError(
            f"SSH_WANNIER_PROBE_FAILED: exit {result.returncode}{suffix}; "
            "verify access in your terminal (password is never stored)"
        )
    stderr = result.stderr or b""
    if len(result.stdout) + len(stderr) > MAX_WANNIER_PROBE_BYTES:
        raise ValueError("WANNIER_PROBE_TOO_LARGE: refusing oversized output")
    snapshot = parse_wannier_probe(
        result.stdout.decode("utf-8", errors="strict"),
        node,
        allowed_nodes=profile.allowed_nodes,
        expected_module=profile.wannier_module,
    )
    return snapshot.model_copy(update={"observed_at": datetime.now(UTC)})


def probe_preview(
    profile: ClusterProfile, node: str, *, interactive: bool = False
) -> dict:
    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    if node not in profile.allowed_nodes:
        raise ValueError("NODE_FORBIDDEN: node is not in the configured cluster profile")
    script = _probe_script(profile, node)
    outer = SSH_OPTIONS_INTERACTIVE if interactive else SSH_OPTIONS
    argv = [
        "ssh", *outer, "-p", str(profile.port),
        f"{profile.user}@{profile.gateway}",
        "/bin/bash -c " + shlex.quote(_gateway_wrapper(profile, node, script)),
    ]
    return {
        "mode": "readonly_probe_preview",
        "node": node,
        "auth_mode": "interactive_password" if interactive else "batch_publickey",
        "ssh_argv": argv,
        "stdin_script": script,
        "stdin_used": False,
        "timeout_seconds": 30,
        "executes_mpi": False,
        "writes_remote_files": False,
        "stores_password": False,
    }


def _cpu_list(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        bounds = part.split("-")
        if len(bounds) not in {1, 2} or not all(x.isascii() and x.isdigit() for x in bounds):
            raise ValueError("invalid CPU affinity list")
        low, high = int(bounds[0]), int(bounds[-1])
        if high < low or high > 1048575 or high - low > 65535:
            raise ValueError("invalid CPU affinity range")
        result.extend(range(low, high + 1))
        if len(result) > 65536:
            raise ValueError("CPU affinity list too large")
    return result


def parse_probe(output: str, node: str, *, allowed_nodes: tuple[str, ...] = ALLOWED_NODES) -> ProbeSnapshot:
    if node not in allowed_nodes or len(output.encode("utf-8")) > MAX_PROBE_BYTES:
        raise ValueError("invalid probe target or oversized output")
    lines = output.splitlines()
    if lines.count("MAGVASP_PROBE_V1") != 1:
        raise ValueError("missing or duplicated probe header")
    # SSH login banners may precede the protocol; no trailing or interleaved data is accepted.
    lines = lines[lines.index("MAGVASP_PROBE_V1") + 1:]
    sections: dict[str, list[str]] = {}
    current = None
    markers = []
    for line in lines:
        if line.startswith("@@"):
            current = line[2:]
            if current in sections:
                raise ValueError("duplicate probe section")
            sections[current] = []
            markers.append(current)
        elif current is None:
            raise ValueError("unexpected data before probe section")
        else:
            sections[current].append(line)
    if markers != [*SECTIONS, "end"] or sections["end"]:
        raise ValueError("incomplete or malformed probe sections")
    try:
        hostname, username = sections["identity"]
        observed = datetime.fromisoformat("".join(sections["time"]))
        cpu_json = json.loads("\n".join(sections["cpus"]))
        cpus = []
        for row in cpu_json["cpus"]:
            online = row["online"]
            if online not in (True, False, "yes", "no") or type(online) not in {bool, str}:
                raise ValueError("unrecognized CPU online flag")
            cpus.append(Cpu(
                cpu=int(row["cpu"]), core=int(row["core"]), socket=int(row["socket"]),
                online=online is True or online == "yes",
            ))
        processes = []
        for line in sections["processes"]:
            pid, user, psr, percent, command = line.split(maxsplit=4)
            processes.append(ProcessObservation(
                pid=int(pid), user=user, last_cpu=int(psr),
                cpu_percent=float(percent), command=command,
            ))
        runtime = dict(line.split("=", 1) for line in sections["runtime"])
        if len(sections["runtime"]) != 5 or set(runtime) != {"module_loaded", "mpi_path", *VASP_PATHS}:
            raise ValueError("unexpected runtime probe fields")
        if any(runtime[key] not in {"yes", "no"} for key in ("module_loaded", *VASP_PATHS)):
            raise ValueError("invalid runtime probe booleans")
        topology_ids = {cpu.cpu for cpu in cpus}
        affinity_ids = _cpu_list("".join(sections["affinity"]))
        # Some nodes expose a wider cgroup affinity mask than the CPUs listed
        # by lscpu (offline/isolated CPUs). Only the intersection is safe to
        # use for MPI binding.
        allowed_ids = [cpu for cpu in affinity_ids if cpu in topology_ids]
        if not allowed_ids:
            raise ValueError("CPU affinity has no CPUs present in topology")
        return ProbeSnapshot(
            source="ssh_probe", observed_at=observed, requested_node=node,
            hostname=hostname, username=username, cpus=cpus,
            allowed_cpu_ids=allowed_ids, processes=processes,
            load1=float(" ".join(sections["load"]).split()[0]),
            mem_available_kib=int("".join(sections["memory"])),
            module_loaded=runtime["module_loaded"] == "yes",
            mpi_path=runtime["mpi_path"] or None,
            executables={name: runtime[name] == "yes" for name in VASP_PATHS},
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed probe payload") from exc


def _ssh_executable() -> str:
    executable = shutil.which("ssh")
    if executable is None:
        raise ValueError("SSH_UNAVAILABLE: install native OpenSSH first")
    return executable


def _run_ssh(argv: list[str], *, timeout: int, interactive: bool) -> subprocess.CompletedProcess:
    """Run a fixed ssh argv. Interactive mode reads the password from the TTY.

    The password is never written to a file, environment variable, or argument
    list. Interactive calls inherit stderr so OpenSSH can prompt; stdout is
    still captured for the typed probe protocol.
    """

    try:
        if interactive:
            # Inherit stdin/stderr so the console can show "password:" and accept typing.
            return subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("SSH_TIMEOUT: command did not finish; no MPI was launched") from exc


def login_preview(profile: ClusterProfile, *, interactive: bool = True) -> dict:
    """One-shot gateway identity check. No compute-node hop, no MPI."""

    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    outer = SSH_OPTIONS_INTERACTIVE if interactive else SSH_OPTIONS
    remote = (
        'set -eu; export LC_ALL=C; '
        f'test "$(hostname -s)" = {shlex.quote(profile.gateway_hostname)} || exit 91; '
        f'test "$(id -un)" = {shlex.quote(profile.user)} || exit 92; '
        'printf "MAGVASP_LOGIN_OK %s %s\\n" "$(hostname -s)" "$(id -un)"'
    )
    argv = [
        "ssh", *outer, "-p", str(profile.port),
        f"{profile.user}@{profile.gateway}", remote,
    ]
    return {
        "mode": "gateway_login_preview",
        "auth_mode": "interactive_password" if interactive else "batch_publickey",
        "ssh_argv": argv,
        "timeout_seconds": 30,
        "executes_mpi": False,
        "writes_remote_files": False,
        "stores_password": False,
    }


def collect_login(profile: ClusterProfile, *, interactive: bool = True) -> dict:
    preview = login_preview(profile, interactive=interactive)
    result = _run_ssh(
        [_ssh_executable(), *preview["ssh_argv"][1:]],
        timeout=preview["timeout_seconds"],
        interactive=interactive,
    )
    if result.returncode:
        raise ValueError(
            f"SSH_LOGIN_FAILED: exit {result.returncode}; "
            "type the password in your own terminal (it is never stored)"
        )
    text = (result.stdout or b"").decode("utf-8", errors="strict")
    marker = f"MAGVASP_LOGIN_OK {profile.gateway_hostname} {profile.user}"
    if marker not in text:
        raise ValueError("SSH_LOGIN_FAILED: gateway identity did not match configured profile")
    return {
        "mode": "gateway_login",
        "ok": True,
        "hostname": profile.gateway_hostname,
        "username": profile.user,
        "auth_mode": preview["auth_mode"],
        "stores_password": False,
    }


def collect_probe(
    profile: ClusterProfile, node: str, *, interactive: bool = False
) -> ProbeSnapshot:
    preview = probe_preview(profile, node, interactive=interactive)
    result = _run_ssh(
        [_ssh_executable(), *preview["ssh_argv"][1:]],
        timeout=preview["timeout_seconds"],
        interactive=interactive,
    )
    if result.returncode:
        # Do not echo arbitrary remote stderr or shell startup data into an agent/tool response.
        stderr = (result.stderr or b"").decode("utf-8", errors="ignore")
        stage = next(
            (line.split(":", 1)[1] for line in stderr.splitlines()
             if line.startswith("MAGVASP_PROBE_ERROR:") and line.count(":") == 1),
            None,
        )
        suffix = f" at fixed stage {stage}" if stage else ""
        raise ValueError(
            f"SSH_PROBE_FAILED: exit {result.returncode}{suffix}; "
            "verify access in your terminal (password is never stored)"
        )
    stderr = result.stderr or b""
    if len(result.stdout) + len(stderr) > MAX_PROBE_BYTES:
        raise ValueError("PROBE_TOO_LARGE: refusing oversized output")
    snapshot = parse_probe(
        result.stdout.decode("utf-8", errors="strict"),
        node,
        allowed_nodes=profile.allowed_nodes,
    )
    # Freshness is a property of when the trusted local controller received the
    # probe, not of the compute node's wall clock.  Cluster clock skew must not
    # make a just-finished probe stale (or future-dated).
    return snapshot.model_copy(update={"observed_at": datetime.now(UTC)})


def assess_resources(
    snapshot: ProbeSnapshot, request: ExecutionRequest, *, now: datetime | None = None,
    profile: ClusterProfile | None = None,
) -> ResourceAssessment:
    snapshot = ProbeSnapshot.model_validate_json(snapshot.model_dump_json())
    request = ExecutionRequest.model_validate_json(request.model_dump_json())
    now = now or datetime.now(UTC)
    if now.utcoffset() is None:
        raise ValueError("assessment time requires a timezone")
    blockers = []

    def block(code: str, message: str) -> None:
        blockers.append(Blocker(code=code, message=message))

    profile = profile or ClusterProfile()
    if request.node not in profile.allowed_nodes:
        block("NODE_FORBIDDEN", "Requested node is not in the configured cluster profile")
    hostname_matches = (
        snapshot.hostname == request.node
        or snapshot.hostname.startswith(f"{request.node}.")
    )
    if snapshot.requested_node != request.node or not hostname_matches or snapshot.username != profile.user:
        block("IDENTITY_MISMATCH", "Observed node/user does not match the requested compute target")
    age = (now - snapshot.observed_at).total_seconds()
    if age < -5 or age > 60:
        block("SNAPSHOT_STALE", "Snapshot is future-dated or older than 60 seconds")
    if snapshot.source != "ssh_probe":
        block("OFFLINE_EVIDENCE", "Offline fixture is not evidence of current cluster availability")
    online = {cpu.cpu: cpu for cpu in snapshot.cpus if cpu.online}
    physical = {(cpu.socket, cpu.core) for cpu in online.values()}
    accessible = {i: cpu for i, cpu in online.items() if i in snapshot.allowed_cpu_ids}
    representatives = {}
    for i, cpu in sorted(accessible.items()):
        representatives.setdefault((cpu.socket, cpu.core), i)
    selected = request.cpu_ids if request.cpu_ids is not None else list(representatives.values())[:request.mpi_ranks]
    if request.mpi_ranks > len(representatives):
        block("INSUFFICIENT_PHYSICAL_CORES", "MPI ranks exceed accessible physical cores, excluding SMT siblings")
    if any(i not in accessible for i in selected):
        block("CPU_NOT_ACCESSIBLE", "Selected CPUs must be online and within the observed affinity")
    keys = [(accessible[i].socket, accessible[i].core) for i in selected if i in accessible]
    if len(keys) != len(set(keys)):
        block("SMT_SIBLING_CONFLICT", "Selected logical CPUs share physical cores")
    active = {
        (online[p.last_cpu].socket, online[p.last_cpu].core)
        for p in snapshot.processes if p.cpu_percent >= 5 and p.last_cpu in online
    }
    if sum(p.cpu_percent for p in snapshot.processes) >= 5 or snapshot.load1 >= 0.5:
        block("NODE_BUSY", "Existing CPU activity or load prevents a non-interference claim; PSR is not a reservation")
    if snapshot.mem_available_kib < 1024 * 1024:
        block("LOW_MEMORY", "Less than 1 GiB MemAvailable; no VASP memory estimate has been verified")
    if not snapshot.module_loaded or snapshot.mpi_path != profile.mpi_path:
        block("RUNTIME_MISMATCH", "Expected VASP module and Intel MPI path were not both observed")
    block("ALLOCATION_UNCONFIRMED", "Only the group/user can grant resource use; an idle snapshot is not an allocation")
    return ResourceAssessment(
        physical_cores=len(physical), logical_cpus=len(online),
        accessible_physical_cores=len(representatives),
        observed_active_physical_cores=len(active), suggested_cpu_ids=selected,
        snapshot_age_seconds=age, blockers=blockers,
    )
