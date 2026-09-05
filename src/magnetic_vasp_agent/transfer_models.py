"""Contracts for transferring an already validated workflow package."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from magnetic_vasp_agent.execution_models import ExecutionModel, Node
from magnetic_vasp_agent.workflow_models import safe_id

TransferSize = Annotated[int, Field(ge=0, le=2**31)]


class TransferRequest(ExecutionModel):
    schema_version: Literal[1] = 1
    node: Node
    run_id: str
    overwrite: Literal[False] = False

    _safe_run = field_validator("run_id")(safe_id)


class TransferFile(ExecutionModel):
    relative_path: str
    size: TransferSize
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        parts = value.replace("\\", "/").split("/")
        if (
            not value
            or value.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
            or any(not part.isascii() for part in parts)
        ):
            raise ValueError("relative_path must be a normalized relative POSIX path")
        return "/".join(parts)


class TransferInventory(ExecutionModel):
    schema_version: Literal[1] = 1
    workflow_id: str
    files: list[TransferFile] = Field(min_length=1, max_length=10000)
    total_bytes: TransferSize


class TransferReport(ExecutionModel):
    schema_version: Literal[1] = 1
    mode: Literal["transfer_plan", "transfer"]
    workflow_id: str
    run_id: str
    node: Node
    local_bundle: str
    remote_directory: str
    files: list[TransferFile]
    ready_to_transfer: bool
    transferred: bool
    remote_hashes_verified: bool
    remote_directory_created: bool
    execution_enabled: Literal[False] = False
    blockers: list[dict[str, str]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
