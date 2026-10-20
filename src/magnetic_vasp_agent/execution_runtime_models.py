"""Typed records for gated launch / status / cancel."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from magnetic_vasp_agent.execution_models import ExecutionModel
from magnetic_vasp_agent.workflow_models import safe_id

SHA256_PATTERN = r"^[0-9a-f]{64}$"
LOCAL_CONFIRM = "I_CONFIRM_LOCAL_LAUNCH"
REMOTE_CONFIRM = "I_CONFIRM_REMOTE_LAUNCH"
SMOKE_CONFIRM = "I_CONFIRM_MPI_SMOKE"


class LaunchBackend(StrEnum):
    LOCAL_SIMULATION = "local_simulation"
    REMOTE_MPI = "remote_mpi"


class JobState(StrEnum):
    BLOCKED = "blocked"
    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETED_UNVERIFIED = "completed_unverified"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class LaunchRequest(ExecutionModel):
    """Caller-supplied launch intent. Confirmation is an exact phrase, not a bool."""

    schema_version: Literal[1] = 1
    backend: LaunchBackend
    confirm: str = Field(min_length=1)
    output: str = Field(min_length=1)
    scenario: Literal["complete", "fail", "timeout", "cancel"] = "complete"
    timeout_seconds: float = Field(default=2.0, ge=0.05, le=10.0)
    run_id: str = "local-sim"
    profile: str | None = None
    bundle: str | None = None
    request: str | None = None
    snapshot: str | None = None
    transfer_report: str | None = None
    smoke_report: str | None = None
    resource_grant: str | None = None
    artifact_release: str | None = None

    _safe_run = field_validator("run_id")(safe_id)


class RunRecord(ExecutionModel):
    """Auditable local record of one launch attempt."""

    schema_version: Literal[1] = 1
    backend: LaunchBackend
    state: JobState
    run_id: str
    output_dir: str
    pid: int | None = None
    exit_code: int | None = None
    events: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    scientific_converged: None = None
    runtime_verified: Literal[False] = False
    cluster_execution: Literal["disabled", "remote_mpi"] = "disabled"
    owned_locally: bool
    command_marker: str | None = None
    node: str | None = None
    remote_directory: str | None = None
    remote_pid: int | None = None
    executable: str | None = None
    record_sha256: str = Field(default="", pattern=r"^([0-9a-f]{64})?$")
