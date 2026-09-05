"""Site-specific, fail-closed contracts for SSH/direct-MPI preparation."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from magnetic_vasp_agent.kpoints_models import StrictModel
from magnetic_vasp_agent.workflow_models import safe_id

Node = Literal["cu21", "cu22", "cu23", "cu41", "cu42", "cu43", "cu48", "cu49", "cu50"]
ALLOWED_NODES = ("cu21", "cu22", "cu23", "cu41", "cu42", "cu43", "cu48", "cu49", "cu50")
MPI_PATH = "/public/intel/oneapi/mpi/2021.1.1/bin/mpirun"
VASP_PATHS = {
    name: f"/public/vasp/vasp.5.4.4/bin/{name}"
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
    gateway: Literal["192.168.164.203"] = "192.168.164.203"
    gateway_hostname: Literal["mu01"] = "mu01"
    user: Literal["zyy"] = "zyy"
    port: Literal[22] = 22
    vasp_module: Literal["vasp/5.4.4-intel"] = "vasp/5.4.4-intel"
    execution_enabled: Literal[False] = False
    remote_root: str | None = None

    @field_validator("execution_enabled", mode="before")
    @classmethod
    def disabled_only(cls, value: object) -> object:
        if value is not False:
            raise ValueError("Step 8 cannot enable remote execution")
        return value

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
