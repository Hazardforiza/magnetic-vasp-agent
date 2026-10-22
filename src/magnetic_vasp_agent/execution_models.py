"""Site-specific, fail-closed contracts for SSH/direct-MPI preparation."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Mapping

from pydantic import ConfigDict, Field, field_validator, model_validator

from magnetic_vasp_agent.kpoints_models import StrictModel
from magnetic_vasp_agent.workflow_models import safe_id

DEFAULT_ALLOWED_NODES = tuple(f"node{index:02d}" for index in range(1, 13))
# Kept as a compatibility alias for callers that used the original policy.
ALLOWED_NODES = DEFAULT_ALLOWED_NODES
Node = str
MPI_PATH = "/opt/intel/oneapi/mpi/bin/mpirun"
WANNIER_MODULE = "vasp/6.4.2-intel-wannier90"
VASP_PATHS = {
    name: f"/opt/vasp/bin/{name}"
    for name in ("vasp_std", "vasp_ncl", "vasp_gam")
}
Index = Annotated[int, Field(ge=0, le=1048575)]


class ExecutionModel(StrictModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, frozen=True
    )


class ClusterProfile(ExecutionModel):
    schema_version: Literal[1] = 1
    transport: Literal["ssh_direct_mpi"] = "ssh_direct_mpi"
    gateway: str = "gateway.example.org"
    gateway_hostname: str = "login01"
    user: str = "cluster_user"
    port: int = Field(22, ge=1, le=65535)
    node_port: int = Field(22, ge=1, le=65535)
    allowed_nodes: tuple[str, ...] = DEFAULT_ALLOWED_NODES
    vasp_module: str = "vasp/6.4.2-intel"
    # Wannier support is a separate VASP build and must be checked explicitly;
    # the ordinary VASP module name cannot establish this capability.
    wannier_module: str = WANNIER_MODULE
    lmod_path: str | None = None
    modulepath: str | None = None
    mpi_path: str = MPI_PATH
    vasp_paths: dict[str, str] = VASP_PATHS
    execution_enabled: bool = False
    remote_root: str | None = None

    @field_validator(
        "gateway", "gateway_hostname", "user", "vasp_module", "wannier_module",
        "mpi_path", "lmod_path",
    )
    @classmethod
    def safe_token(cls, value: str) -> str:
        if value is None:
            return value
        if not value or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("cluster profile fields must be nonempty shell-safe tokens")
        if any(c in value for c in "'\";|&$`<>\\\n"):
            raise ValueError("cluster profile fields contain forbidden shell characters")
        return value

    @field_validator("mpi_path")
    @classmethod
    def absolute_mpi_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("mpi_path must be an absolute executable path")
        return value

    @field_validator("lmod_path")
    @classmethod
    def absolute_lmod_path(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("/"):
            raise ValueError("lmod_path must be an absolute executable path")
        return value

    @field_validator("modulepath")
    @classmethod
    def safe_modulepath(cls, value: str | None) -> str | None:
        if value is not None and (
            not value or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(c in value for c in "'\";|&$`<>\\\n")
        ):
            raise ValueError("modulepath must be a safe colon-separated path list")
        return value

    @field_validator("allowed_nodes")
    @classmethod
    def safe_nodes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("allowed_nodes must be a non-empty unique list")
        for node in value:
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", node) is None:
                raise ValueError("allowed_nodes contains an unsafe hostname")
        return value

    @field_validator("vasp_paths")
    @classmethod
    def safe_vasp_paths(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != set(VASP_PATHS):
            raise ValueError("vasp_paths must define vasp_std, vasp_ncl and vasp_gam")
        for path in value.values():
            if not path.startswith("/") or any(c.isspace() or c in "'\";|&$`<>\\" for c in path):
                raise ValueError("vasp executable paths must be absolute shell-safe paths")
        return value

    @model_validator(mode="after")
    def gateway_is_jump_only(self) -> "ClusterProfile":
        if self.gateway_hostname in self.allowed_nodes:
            raise ValueError("gateway hostname must not be a compute node")
        return self

    @model_validator(mode="after")
    def enabled_requires_remote_root(self) -> "ClusterProfile":
        if self.execution_enabled and self.remote_root is None:
            raise ValueError("execution_enabled requires a dedicated remote_root")
        return self

    @field_validator("remote_root")
    @classmethod
    def safe_remote_root(cls, value: str | None) -> str | None:
        if value is None:
            return value
        # Narrow ASCII paths only, never a home root, relative path or shell fragment.
        parts = value.split("/")
        if (
            not value.startswith("/")
            or len(parts) < 5
            or len(value) > 512
            or any(
                part in {".", ".."}
                or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part) is None
                for part in parts[1:]
            )
        ):
            raise ValueError("remote_root must be a dedicated absolute POSIX subdirectory")
        return value


class ExecutionRequest(ExecutionModel):
    schema_version: Literal[1] = 1
    node: Node
    step_id: str
    run_id: str
    mpi_ranks: Annotated[int, Field(ge=1, le=1024)] = 1
    cpu_ids: list[Index] | None = None

    _safe_names = field_validator("step_id", "run_id")(safe_id)

    @field_validator("node")
    @classmethod
    def safe_node(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value) is None:
            raise ValueError("node must be a safe hostname token")
        return value

    @model_validator(mode="after")
    def explicit_cpu_count(self) -> ExecutionRequest:
        if self.cpu_ids is not None and (
            len(self.cpu_ids) != self.mpi_ranks
            or len(set(self.cpu_ids)) != len(self.cpu_ids)
        ):
            raise ValueError("cpu_ids must contain one distinct logical CPU per MPI rank")
        return self


class Cpu(ExecutionModel):
    cpu: Index
    core: Index
    socket: Index
    online: bool


class ProcessObservation(ExecutionModel):
    pid: Annotated[int, Field(gt=0)]
    user: Annotated[str, Field(min_length=1, max_length=128)]
    last_cpu: Index
    cpu_percent: Annotated[float, Field(ge=0)]
    command: Annotated[str, Field(min_length=1, max_length=128)]

    @field_validator("user", "command")
    @classmethod
    def no_controls(cls, value: str) -> str:
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("process fields must not contain control characters")
        return value


class ProbeSnapshot(ExecutionModel):
    schema_version: Literal[1] = 1
    source: Literal["ssh_probe", "offline_fixture"]
    observed_at: datetime
    requested_node: Node
    hostname: str
    username: str
    cpus: Annotated[list[Cpu], Field(min_length=1, max_length=65536)]
    allowed_cpu_ids: Annotated[list[Index], Field(min_length=1, max_length=65536)]
    processes: Annotated[list[ProcessObservation], Field(max_length=100000)]
    load1: Annotated[float, Field(ge=0)]
    mem_available_kib: Annotated[int, Field(ge=0)]
    module_loaded: bool
    mpi_path: str | None
    executables: dict[Literal["vasp_std", "vasp_ncl", "vasp_gam"], bool]

    @field_validator("observed_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("snapshot timestamp requires a timezone")
        return value

    @model_validator(mode="after")
    def topology_consistent(self) -> ProbeSnapshot:
        ids = [cpu.cpu for cpu in self.cpus]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate CPU IDs in topology")
        if len(self.allowed_cpu_ids) != len(set(self.allowed_cpu_ids)):
            raise ValueError("duplicate allowed CPU IDs")
        if not set(self.allowed_cpu_ids).issubset(ids):
            raise ValueError("allowed CPU IDs are missing from topology")
        if len({p.pid for p in self.processes}) != len(self.processes):
            raise ValueError("duplicate process PIDs")
        if any(p.last_cpu not in ids for p in self.processes):
            raise ValueError("process CPU is missing from topology")
        if set(self.executables) != set(VASP_PATHS):
            raise ValueError("snapshot must check all three known VASP executable paths")
        return self


class WannierProbeSnapshot(ExecutionModel):
    """Read-only evidence for the VASP/Wannier90 runtime on one compute node.

    This is intentionally separate from ``ProbeSnapshot``.  A normal VASP
    module probe must never imply that the Wannier interface was compiled in.
    Paths are reported by the module-selected ``command -v`` result and the
    version is extracted from the fixed ``wannier90.x -v`` banner.
    """

    schema_version: Literal[1] = 1
    source: Literal["ssh_wannier_probe", "offline_fixture"]
    observed_at: datetime
    requested_node: Node
    hostname: str
    username: str
    module_name: str
    module_loaded: bool
    wannier90_path: str | None
    wannier90_version: str | None
    vasp_ncl_path: str | None

    @field_validator("observed_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("Wannier snapshot timestamp requires a timezone")
        return value

    @field_validator("module_name")
    @classmethod
    def safe_module_name(cls, value: str) -> str:
        if not value or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127
            for char in value
        ) or any(char in value for char in "'\";|&$`<>\\"):
            raise ValueError("Wannier module name must be a shell-safe token")
        return value

    @field_validator("hostname", "username")
    @classmethod
    def safe_identity(cls, value: str) -> str:
        if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("Wannier probe identity contains control characters")
        return value

    @field_validator("wannier90_path", "vasp_ncl_path")
    @classmethod
    def safe_executable_path(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if (
            not value.startswith("/")
            or len(value) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or any(char in value for char in "'\";|&$`<>\\")
        ):
            raise ValueError("Wannier executable paths must be absolute shell-safe paths")
        return value

    @field_validator("wannier90_version")
    @classmethod
    def safe_version(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) > 128
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("Wannier version contains invalid control characters")
        return value

    @model_validator(mode="after")
    def module_contract(self) -> "WannierProbeSnapshot":
        if self.module_loaded and self.wannier90_path is None:
            # A loaded module with no discoverable executable is useful
            # evidence, but it cannot be treated as a usable capability.
            return self
        if self.wannier90_version is not None and self.wannier90_path is None:
            raise ValueError("Wannier version requires a discovered wannier90.x path")
        return self

    @property
    def capability_verified(self) -> bool:
        """Whether all runtime markers needed for a Wannier handoff exist."""

        return bool(
            self.module_loaded
            and self.wannier90_path
            and self.wannier90_version
            and self.vasp_ncl_path
        )


class Blocker(ExecutionModel):
    code: str
    message: str


class ResourceAssessment(ExecutionModel):
    physical_cores: int
    logical_cpus: int
    accessible_physical_cores: int
    observed_active_physical_cores: int
    suggested_cpu_ids: list[int]
    snapshot_age_seconds: float
    blockers: list[Blocker]
    allocation_verified: Literal[False] = False
    ready_to_run: Literal[False] = False


# Resource grants and the execution gate below are deliberately separate from
# ``ProbeSnapshot`` and ``ResourceAssessment``.  A snapshot is an observation
# of a shared machine; it can never become an allocation merely by being recent
# or idle.  These contracts are local-only and do not open an SSH connection.
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
InputName = Literal["POSCAR", "POTCAR", "INCAR", "KPOINTS"]


def _safe_text(value: str, *, field: str) -> str:
    if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field} must be nonempty text without control characters")
    return value


class ResourceGrant(ExecutionModel):
    """A human-issued, time-bounded allocation for one run.

    The grant is an independent approval artifact.  It binds the exact
    workflow/step/run and input hashes to one node and one CPU set; a probe
    result cannot satisfy any of these fields.
    """

    schema_version: Literal[1] = 1
    grant_id: str = "grant"
    workflow_id: str
    step_id: str
    run_id: str
    workflow_manifest_sha256: SHA256
    step_manifest_sha256: SHA256
    run_sha256: SHA256
    input_sha256: dict[InputName, SHA256]
    node: Node
    mpi_ranks: Annotated[int, Field(ge=1, le=1024)]
    cpu_ids: Annotated[list[Index], Field(min_length=1, max_length=1024)]
    valid_from: datetime
    valid_until: datetime
    exclusive: bool = True
    memory_limit_kib: Annotated[int, Field(gt=0)] | None = None
    approver: str
    approved_at: datetime

    _safe_ids = field_validator("grant_id", "workflow_id", "step_id", "run_id")(
        safe_id
    )

    @field_validator("approver")
    @classmethod
    def named_approver(cls, value: str) -> str:
        return _safe_text(value, field="approver")

    @field_validator("valid_from", "valid_until", "approved_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("resource grant timestamps require a timezone")
        return value

    @field_validator("input_sha256")
    @classmethod
    def complete_input_inventory(cls, value: dict[InputName, SHA256]) -> dict[InputName, SHA256]:
        expected = {"POSCAR", "POTCAR", "INCAR", "KPOINTS"}
        if set(value) != expected:
            raise ValueError("resource grant must bind all four input hashes")
        return value

    @model_validator(mode="after")
    def grant_consistent(self) -> "ResourceGrant":
        if len(self.cpu_ids) != self.mpi_ranks or len(set(self.cpu_ids)) != len(self.cpu_ids):
            raise ValueError("resource grant requires one distinct CPU per MPI rank")
        if self.valid_until <= self.valid_from:
            raise ValueError("resource grant valid_until must be after valid_from")
        if not self.valid_from <= self.approved_at <= self.valid_until:
            raise ValueError("resource grant approval must fall inside its validity window")
        return self

    @property
    def workflow_sha256(self) -> str:
        """Short alias used by callers that do not need the manifest suffix."""

        return self.workflow_manifest_sha256

    @property
    def step_sha256(self) -> str:
        return self.step_manifest_sha256

    def covers(self, now: datetime) -> bool:
        if now.utcoffset() is None:
            raise ValueError("resource grant check requires a timezone")
        return self.valid_from <= now < self.valid_until

    def matches_request(self, request: ExecutionRequest) -> bool:
        return (
            self.workflow_id is not None
            and self.step_id == request.step_id
            and self.run_id == request.run_id
            and self.node == request.node
            and self.mpi_ranks == request.mpi_ranks
            and (request.cpu_ids is None or self.cpu_ids == request.cpu_ids)
        )

    def binds_artifacts(
        self,
        *,
        workflow_id: str,
        step_id: str,
        run_id: str,
        workflow_manifest_sha256: str,
        step_manifest_sha256: str,
        run_sha256: str,
        input_sha256: Mapping[str, str],
    ) -> bool:
        return (
            self.workflow_id == workflow_id
            and self.step_id == step_id
            and self.run_id == run_id
            and self.workflow_manifest_sha256 == workflow_manifest_sha256
            and self.step_manifest_sha256 == step_manifest_sha256
            and self.run_sha256 == run_sha256
            and dict(self.input_sha256) == dict(input_sha256)
        )


class SmokeTestEvidence(ExecutionModel):
    """Result of the fixed, non-scientific single-node MPI smoke test."""

    schema_version: Literal[1] = 1
    run_id: str
    node: Node
    mpi_ranks: Annotated[int, Field(ge=1, le=1024)]
    cpu_ids: Annotated[list[Index], Field(min_length=1, max_length=1024)]
    hostname: str
    started_at: datetime
    finished_at: datetime
    exit_code: int
    timed_out: bool = False
    rank_cpu_ids: Annotated[list[Index], Field(min_length=1, max_length=1024)]
    stdout_sha256: SHA256
    stderr_sha256: SHA256
    output_bytes: Annotated[int, Field(ge=0)] = 0
    passed: bool

    _safe_run = field_validator("run_id")(safe_id)

    @field_validator("hostname")
    @classmethod
    def hostname_safe(cls, value: str) -> str:
        value = _safe_text(value, field="hostname")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", value):
            raise ValueError("smoke-test hostname has an unsafe format")
        return value

    @field_validator("started_at", "finished_at")
    @classmethod
    def smoke_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("smoke-test timestamps require a timezone")
        return value

    @model_validator(mode="after")
    def smoke_consistent(self) -> "SmokeTestEvidence":
        if self.finished_at <= self.started_at:
            raise ValueError("smoke-test finished_at must be after started_at")
        if len(self.cpu_ids) != self.mpi_ranks or len(set(self.cpu_ids)) != len(self.cpu_ids):
            raise ValueError("smoke-test requires one distinct approved CPU per rank")
        if len(self.rank_cpu_ids) != self.mpi_ranks:
            raise ValueError("rank_cpu_ids must contain one CPU per MPI rank")
        if any(cpu not in self.cpu_ids for cpu in self.rank_cpu_ids):
            raise ValueError("smoke-test rank CPU is outside the approved CPU set")
        if len(set(self.rank_cpu_ids)) != len(self.rank_cpu_ids):
            raise ValueError("smoke-test rank CPU IDs must be distinct")
        if not (self.hostname == self.node or self.hostname.startswith(f"{self.node}.")):
            raise ValueError("smoke-test ran on an unexpected host")
        expected = self.exit_code == 0 and not self.timed_out
        if self.passed != expected:
            raise ValueError("smoke-test passed must agree with timeout and exit code")
        return self


class TransferVerification(ExecutionModel):
    """Hash-only proof that the approved input bundle reached its new leaf dir."""

    schema_version: Literal[1] = 1
    run_id: str
    remote_directory: str
    files: dict[str, SHA256]
    report_sha256: SHA256
    verified: bool
    verified_at: datetime

    _safe_run = field_validator("run_id")(safe_id)

    @field_validator("remote_directory")
    @classmethod
    def safe_directory(cls, value: str) -> str:
        if (
            not value.startswith("/")
            or len(value) > 1024
            or any(
                part in {"", ".", ".."}
                or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part) is None
                for part in value.split("/")[1:]
            )
        ):
            raise ValueError("transfer directory must be an absolute dedicated POSIX path")
        return value

    @field_validator("verified_at")
    @classmethod
    def transfer_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("transfer timestamp requires a timezone")
        return value

    @field_validator("files")
    @classmethod
    def transfer_inventory(cls, value: dict[str, SHA256]) -> dict[str, SHA256]:
        if not value:
            raise ValueError("transfer verification requires a nonempty file inventory")
        if any("POTCAR" not in name and name.endswith(("OUTCAR", "CHGCAR", "WAVECAR", "vasprun.xml")) for name in value):
            raise ValueError("transfer inventory contains an unapproved output file")
        return value


class LaunchApproval(ExecutionModel):
    """A second, per-run human approval immediately before VASP launch."""

    schema_version: Literal[1] = 1
    run_id: str
    workflow_manifest_sha256: SHA256
    step_manifest_sha256: SHA256
    run_sha256: SHA256
    smoke_test_sha256: SHA256
    transfer_report_sha256: SHA256
    approver: str
    approved_at: datetime
    valid_until: datetime

    _safe_run = field_validator("run_id")(safe_id)

    @field_validator("approver")
    @classmethod
    def approval_name(cls, value: str) -> str:
        return _safe_text(value, field="approver")

    @field_validator("approved_at", "valid_until")
    @classmethod
    def approval_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("launch approval timestamps require a timezone")
        return value

    @model_validator(mode="after")
    def approval_window(self) -> "LaunchApproval":
        if self.valid_until <= self.approved_at:
            raise ValueError("launch approval valid_until must be after approved_at")
        return self

    def covers(self, now: datetime) -> bool:
        if now.utcoffset() is None:
            raise ValueError("launch approval check requires a timezone")
        return self.approved_at <= now < self.valid_until


class ExecutionStage(StrEnum):
    """Ordered stages for one run; values are stable audit vocabulary."""

    VALIDATED_INPUTS = "validated_inputs"
    HUMAN_RESOURCE_GRANT = "human_resource_grant"
    FRESH_PROBE = "fresh_probe"
    FIXED_SMOKE_TEST = "fixed_smoke_test"
    HUMAN_LAUNCH_APPROVAL = "human_launch_approval"
    TRANSFER_GUARD = "transfer_guard"
    TRANSFERRED_VERIFIED = "transferred_verified"
    LAUNCH = "launch"
    READONLY_STATUS = "readonly_status"
    REVIEWED_RESULT = "reviewed_result"
    BLOCKED = "blocked"
    FAILED_MANUAL_REVIEW = "failed_manual_review"


class ExecutionState(ExecutionModel):
    """Serializable state and evidence pointers for one execution attempt."""

    schema_version: Literal[1] = 1
    mode: Literal["offline_execution_design"] = "offline_execution_design"
    execution_enabled: Literal[False] = False
    ready_to_run: Literal[False] = False
    approval_authenticity_verified: Literal[False] = False
    workflow_id: str
    step_id: str
    run_id: str
    stage: ExecutionStage = ExecutionStage.VALIDATED_INPUTS
    blockers: list[str] = Field(default_factory=list)
    resource_grant: ResourceGrant | None = None
    snapshot_sha256: SHA256 | None = None
    smoke_test_sha256: SHA256 | None = None
    transfer_report_sha256: SHA256 | None = None
    launch_approval: LaunchApproval | None = None

    _safe_state_ids = field_validator("workflow_id", "step_id", "run_id")(safe_id)


_STAGE_NEXT: dict[ExecutionStage, ExecutionStage] = {
    ExecutionStage.VALIDATED_INPUTS: ExecutionStage.HUMAN_RESOURCE_GRANT,
    ExecutionStage.HUMAN_RESOURCE_GRANT: ExecutionStage.FRESH_PROBE,
    ExecutionStage.FRESH_PROBE: ExecutionStage.FIXED_SMOKE_TEST,
    ExecutionStage.FIXED_SMOKE_TEST: ExecutionStage.HUMAN_LAUNCH_APPROVAL,
    ExecutionStage.HUMAN_LAUNCH_APPROVAL: ExecutionStage.TRANSFER_GUARD,
    ExecutionStage.TRANSFER_GUARD: ExecutionStage.TRANSFERRED_VERIFIED,
    ExecutionStage.TRANSFERRED_VERIFIED: ExecutionStage.LAUNCH,
    ExecutionStage.LAUNCH: ExecutionStage.READONLY_STATUS,
    ExecutionStage.READONLY_STATUS: ExecutionStage.REVIEWED_RESULT,
}


def _model_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


class ExecutionStateMachine:
    """Fail-closed local state machine for the real-cluster design.

    Methods only record evidence.  They deliberately contain no transport or
    process-launch code.  A failed smoke test, stale probe, expired grant or
    changed hash moves the run to ``blocked`` and cannot be bypassed.
    """

    def __init__(
        self,
        workflow_id: str,
        step_id: str,
        run_id: str,
        *,
        workflow_manifest_sha256: str | None = None,
        step_manifest_sha256: str | None = None,
        run_sha256: str | None = None,
        input_sha256: Mapping[str, str] | None = None,
        max_snapshot_age_seconds: float = 60.0,
    ) -> None:
        self.state = ExecutionState(workflow_id=workflow_id, step_id=step_id, run_id=run_id)
        self._hashes: dict[str, object] = {
            "workflow_manifest_sha256": workflow_manifest_sha256,
            "step_manifest_sha256": step_manifest_sha256,
            "run_sha256": run_sha256,
            "input_sha256": dict(input_sha256) if input_sha256 is not None else None,
        }
        self.max_snapshot_age_seconds = max_snapshot_age_seconds

    @property
    def stage(self) -> ExecutionStage:
        return self.state.stage

    def _block(self, reason: str) -> ExecutionState:
        self.state = self.state.model_copy(
            update={"stage": ExecutionStage.BLOCKED, "blockers": [*self.state.blockers, reason]}
        )
        return self.state

    def _require(self, expected: ExecutionStage) -> None:
        if self.state.stage != expected:
            raise ValueError(f"invalid execution transition: expected {expected.value}, got {self.state.stage.value}")

    def _bind_hashes(self, grant: ResourceGrant) -> None:
        values = {
            "workflow_manifest_sha256": grant.workflow_manifest_sha256,
            "step_manifest_sha256": grant.step_manifest_sha256,
            "run_sha256": grant.run_sha256,
            "input_sha256": dict(grant.input_sha256),
        }
        for key, value in values.items():
            existing = self._hashes.get(key)
            if existing is not None and existing != value:
                raise ValueError(f"{key} does not match the state identity")
            self._hashes[key] = value

    def record_resource_grant(
        self, grant: ResourceGrant, request: ExecutionRequest, *, now: datetime
    ) -> ExecutionState:
        self._require(ExecutionStage.VALIDATED_INPUTS)
        if (
            grant.workflow_id != self.state.workflow_id
            or not grant.matches_request(request)
            or request.step_id != self.state.step_id
            or request.run_id != self.state.run_id
        ):
            raise ValueError("resource grant does not match this execution request")
        if not grant.covers(now):
            return self._block("RESOURCE_GRANT_EXPIRED")
        self._bind_hashes(grant)
        self.state = self.state.model_copy(
            update={"stage": ExecutionStage.HUMAN_RESOURCE_GRANT, "resource_grant": grant}
        )
        return self.state

    def record_probe(
        self,
        snapshot: ProbeSnapshot,
        request: ExecutionRequest,
        *,
        now: datetime,
        snapshot_sha256: str | None = None,
        profile: ClusterProfile | None = None,
    ) -> ExecutionState:
        self._require(ExecutionStage.HUMAN_RESOURCE_GRANT)
        grant = self.state.resource_grant
        if grant is None or not grant.covers(now):
            return self._block("RESOURCE_GRANT_EXPIRED")
        profile = profile or ClusterProfile()
        if request.node not in profile.allowed_nodes or grant.node not in profile.allowed_nodes:
            return self._block("NODE_FORBIDDEN")
        if snapshot.requested_node != request.node or snapshot.requested_node != grant.node:
            return self._block("PROBE_NODE_MISMATCH")
        if not (
            snapshot.hostname == snapshot.requested_node
            or snapshot.hostname.startswith(f"{snapshot.requested_node}.")
        ) or snapshot.username != profile.user:
            return self._block("IDENTITY_MISMATCH")
        if not set(grant.cpu_ids).issubset(snapshot.allowed_cpu_ids):
            return self._block("CPU_NOT_ACCESSIBLE")
        age = (now - snapshot.observed_at).total_seconds()
        # The probe timestamp is produced remotely; tolerate only the same
        # bounded five-second clock skew used by ``assess_resources``.
        if age < -5 or age > self.max_snapshot_age_seconds:
            return self._block("SNAPSHOT_STALE")
        digest = snapshot_sha256 or _model_digest(snapshot.model_dump(mode="json"))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("snapshot_sha256 must be a lowercase SHA-256")
        self.state = self.state.model_copy(
            update={"stage": ExecutionStage.FRESH_PROBE, "snapshot_sha256": digest}
        )
        return self.state

    def record_smoke_test(self, evidence: SmokeTestEvidence) -> ExecutionState:
        self._require(ExecutionStage.FRESH_PROBE)
        grant = self.state.resource_grant
        if grant is None or evidence.run_id != self.state.run_id:
            raise ValueError("smoke-test evidence is not bound to this run")
        if (
            evidence.node != grant.node
            or evidence.mpi_ranks != grant.mpi_ranks
            or evidence.cpu_ids != grant.cpu_ids
        ):
            raise ValueError("smoke-test evidence does not match the resource grant")
        if not evidence.passed:
            return self._block("MPI_SMOKE_TEST_FAILED")
        digest = _model_digest(evidence.model_dump(mode="json"))
        self.state = self.state.model_copy(
            update={"stage": ExecutionStage.FIXED_SMOKE_TEST, "smoke_test_sha256": digest}
        )
        return self.state

    def approve_launch(self, approval: LaunchApproval, *, now: datetime) -> ExecutionState:
        self._require(ExecutionStage.FIXED_SMOKE_TEST)
        grant = self.state.resource_grant
        if grant is None or approval.run_id != self.state.run_id:
            raise ValueError("launch approval is not bound to this run")
        if not approval.covers(now):
            return self._block("LAUNCH_APPROVAL_EXPIRED")
        if (
            approval.workflow_manifest_sha256 != grant.workflow_manifest_sha256
            or approval.step_manifest_sha256 != grant.step_manifest_sha256
            or approval.run_sha256 != grant.run_sha256
            or approval.smoke_test_sha256 != self.state.smoke_test_sha256
        ):
            return self._block("LAUNCH_APPROVAL_HASH_MISMATCH")
        self.state = self.state.model_copy(
            update={"stage": ExecutionStage.HUMAN_LAUNCH_APPROVAL, "launch_approval": approval}
        )
        return self.state

    def begin_transfer(self) -> ExecutionState:
        self._require(ExecutionStage.HUMAN_LAUNCH_APPROVAL)
        self.state = self.state.model_copy(update={"stage": ExecutionStage.TRANSFER_GUARD})
        return self.state

    def record_transfer(self, verification: TransferVerification) -> ExecutionState:
        self._require(ExecutionStage.TRANSFER_GUARD)
        if verification.run_id != self.state.run_id or not verification.verified:
            return self._block("TRANSFER_NOT_VERIFIED")
        self.state = self.state.model_copy(
            update={
                "stage": ExecutionStage.TRANSFERRED_VERIFIED,
                "transfer_report_sha256": verification.report_sha256,
            }
        )
        return self.state

    def launch(self) -> ExecutionState:
        self._require(ExecutionStage.TRANSFERRED_VERIFIED)
        self.state = self.state.model_copy(update={"stage": ExecutionStage.LAUNCH})
        return self.state

    def record_status(self) -> ExecutionState:
        self._require(ExecutionStage.LAUNCH)
        self.state = self.state.model_copy(update={"stage": ExecutionStage.READONLY_STATUS})
        return self.state

    def review_result(self) -> ExecutionState:
        self._require(ExecutionStage.READONLY_STATUS)
        self.state = self.state.model_copy(update={"stage": ExecutionStage.REVIEWED_RESULT})
        return self.state
