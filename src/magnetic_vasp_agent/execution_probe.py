"""Fixed read-only probe. No upload, MPI launch, process signals or arbitrary commands."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from datetime import UTC, datetime

from magnetic_vasp_agent.execution_models import (
    ALLOWED_NODES,
    MPI_PATH,
    VASP_PATHS,
    Blocker,
    ClusterProfile,
    Cpu,
    ExecutionRequest,
    ProbeSnapshot,
    ProcessObservation,
    ResourceAssessment,
)

SECTIONS = ("identity", "time", "cpus", "affinity", "processes", "load", "memory", "runtime")
MAX_PROBE_BYTES = 2_000_000
SSH_OPTIONS = [
    "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
    "-o", "PermitLocalCommand=no", "-o", "ConnectTimeout=8",
    "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
    "-o", "ControlMaster=no", "-o", "ControlPath=none",
]


def probe_preview(profile: ClusterProfile, node: str) -> dict:
    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    if node not in ALLOWED_NODES:
        raise ValueError("NODE_FORBIDDEN: only the nine group compute nodes are allowed")
    inner = ["ssh", *SSH_OPTIONS, "-p", "22", f"{profile.user}@{node}", "/bin/bash -s"]
    # Both hops have their own strict options; the gateway only checks identity and forwards stdin.
    gateway = (
        'test "$(hostname -s)" = mu01 || exit 91\n'
        'test "$(id -un)" = zyy || exit 92\n'
        "exec " + shlex.join(inner)
    )
    argv = [
        "ssh", *SSH_OPTIONS, "-p", str(profile.port),
        f"{profile.user}@{profile.gateway}", "/bin/bash -c " + shlex.quote(gateway),
    ]
    script = f"""set -eu
export LC_ALL=C
test "$(hostname -s)" = {node} || exit 91
test "$(id -un)" = zyy || exit 92
printf 'MAGVASP_PROBE_V1\\n@@identity\\n'
hostname
id -un
printf '@@time\\n'
date -u +%Y-%m-%dT%H:%M:%SZ
printf '@@cpus\\n'
lscpu --json --extended=CPU,CORE,SOCKET,ONLINE
printf '@@affinity\\n'
awk '/^Cpus_allowed_list:/ {{print $2}}' /proc/self/status
printf '@@processes\\n'
ps -eo pid=,user=,psr=,pcpu=,comm=
printf '@@load\\n'
cat /proc/loadavg
printf '@@memory\\n'
awk '/^MemAvailable:/ {{print $2}}' /proc/meminfo
printf '@@runtime\\n'
if ! type module >/dev/null 2>&1; then
    if test -r /etc/profile.d/lmod.sh; then
        source /etc/profile.d/lmod.sh
    elif test -r /etc/profile.d/modules.sh; then
        source /etc/profile.d/modules.sh
    fi
fi
if type module >/dev/null 2>&1 && module load {profile.vasp_module} >/dev/null 2>&1; then
    printf 'module_loaded=yes\\n'
else
    printf 'module_loaded=no\\n'
fi
printf 'mpi_path=%s\\n' "$(command -v mpirun || true)"
"""
    for name, path in VASP_PATHS.items():
        script += f"if test -x {path}; then printf '{name}=yes\\n'; else printf '{name}=no\\n'; fi\n"
    script += "printf '@@end\\n'\n"
    return {
        "mode": "readonly_probe_preview", "node": node, "ssh_argv": argv,
        "stdin_script": script, "timeout_seconds": 30,
        "executes_mpi": False, "writes_remote_files": False,
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


def parse_probe(output: str, node: str) -> ProbeSnapshot:
    if node not in ALLOWED_NODES or len(output.encode("utf-8")) > MAX_PROBE_BYTES:
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
        return ProbeSnapshot(
            source="ssh_probe", observed_at=observed, requested_node=node,
            hostname=hostname, username=username, cpus=cpus,
            allowed_cpu_ids=_cpu_list("".join(sections["affinity"])), processes=processes,
            load1=float(" ".join(sections["load"]).split()[0]),
            mem_available_kib=int("".join(sections["memory"])),
            module_loaded=runtime["module_loaded"] == "yes",
            mpi_path=runtime["mpi_path"] or None,
            executables={name: runtime[name] == "yes" for name in VASP_PATHS},
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed probe payload") from exc


def collect_probe(profile: ClusterProfile, node: str) -> ProbeSnapshot:
    preview = probe_preview(profile, node)
    executable = shutil.which("ssh")
    if executable is None:
        raise ValueError("SSH_UNAVAILABLE: install native OpenSSH first")
    try:
        result = subprocess.run(
            [executable, *preview["ssh_argv"][1:]],
            input=preview["stdin_script"].encode("utf-8"),
            capture_output=True, timeout=preview["timeout_seconds"], check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("SSH_TIMEOUT: read-only probe did not finish; no MPI was launched") from exc
    if result.returncode:
        # Do not echo arbitrary remote stderr or shell startup data into an agent/tool response.
        raise ValueError(f"SSH_PROBE_FAILED: exit {result.returncode}; verify access in your terminal")
    if len(result.stdout) + len(result.stderr) > MAX_PROBE_BYTES:
        raise ValueError("PROBE_TOO_LARGE: refusing oversized output")
    return parse_probe(result.stdout.decode("utf-8", errors="strict"), node)


def assess_resources(
    snapshot: ProbeSnapshot, request: ExecutionRequest, *, now: datetime | None = None
) -> ResourceAssessment:
    snapshot = ProbeSnapshot.model_validate_json(snapshot.model_dump_json())
    request = ExecutionRequest.model_validate_json(request.model_dump_json())
    now = now or datetime.now(UTC)
    if now.utcoffset() is None:
        raise ValueError("assessment time requires a timezone")
    blockers = []

    def block(code: str, message: str) -> None:
        blockers.append(Blocker(code=code, message=message))

    if snapshot.requested_node != request.node or snapshot.hostname not in {
        request.node, f"{request.node}.cmphc",
    } or snapshot.username != "zyy":
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
    if not snapshot.module_loaded or snapshot.mpi_path != MPI_PATH:
        block("RUNTIME_MISMATCH", "Expected VASP module and Intel MPI path were not both observed")
    block("ALLOCATION_UNCONFIRMED", "Only the group/user can grant resource use; an idle snapshot is not an allocation")
    return ResourceAssessment(
        physical_cores=len(physical), logical_cpus=len(online),
        accessible_physical_cores=len(representatives),
        observed_active_physical_cores=len(active), suggested_cpu_ids=selected,
        snapshot_age_seconds=age, blockers=blockers,
    )
