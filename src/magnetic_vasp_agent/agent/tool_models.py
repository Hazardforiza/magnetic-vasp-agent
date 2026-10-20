"""Typed, auditable protocol for the local VASP agent tool layer.

Every model here is a strict Pydantic model with no external dependency, so the
protocol, approval gate, and runner remain fully testable offline. Provenance is
recorded with canonical-JSON SHA256 digests, mirroring the deterministic core.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CALL_ID_PATTERN = r"^T[1-9][0-9]*$"
CONTROLLED_CLUSTER_EXECUTE_TOOLS = frozenset(
    {
        "execution.smoke",
        "execution.transfer",
        "execution.launch",
        "execution.artifact_release",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic UTF-8 JSON used for every recorded hash in this layer."""

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_of(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class RiskTier(StrEnum):
    """How much a tool can affect the world; the approval gate keys off this."""

    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    CLUSTER_PLAN = "cluster_plan"
    CLUSTER_EXECUTE = "cluster_execute"


class ApprovalState(StrEnum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class ToolSpec(StrictModel):
    """Public description of a registered tool; safe to expose to an LLM."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    risk_tier: RiskTier
    produces_files: bool
    input_schema: dict[str, Any]


class ToolCall(StrictModel):
    """A single proposed tool invocation with typed arguments."""

    call_id: str = Field(pattern=CALL_ID_PATTERN)
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ApprovalRecord(StrictModel):
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    tool: str
    risk_tier: RiskTier
    state: ApprovalState
    reason: str | None = None
    approver: str | None = None


class ToolResult(StrictModel):
    call_id: str = Field(pattern=CALL_ID_PATTERN)
    tool: str
    risk_tier: RiskTier
    approval_state: ApprovalState
    status: Literal["ok", "validation_error", "blocked", "error"]
    output: dict[str, Any] | None = None
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    message: str | None = None

    @model_validator(mode="after")
    def output_hash_contract(self) -> ToolResult:
        if self.output is None:
            if self.output_sha256 is not None:
                raise ValueError("output_sha256 requires an output payload")
        else:
            expected = sha256_of(self.output)
            if self.output_sha256 != expected:
                raise ValueError("output_sha256 does not match output payload")
        return self


class AgentPlan(StrictModel):
    """An ordered list of typed tool calls; produced by hand or by the LLM."""

    schema_version: Literal[1] = 1
    goal: str = Field(min_length=1)
    calls: list[ToolCall] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_call_ids(self) -> AgentPlan:
        ids = [call.call_id for call in self.calls]
        if len(ids) != len(set(ids)):
            raise ValueError("agent plan call ids must be unique")
        return self

    def digest(self) -> str:
        return sha256_of(self.model_dump(mode="json"))


class AgentTranscript(StrictModel):
    """Auditable record of a run, including narrowly gated cluster actions.

    Cluster execution remains disabled unless a fixed controlled tool completed
    after an explicit, named approval. Human review is always required before
    any result is treated as final.
    """

    schema_version: Literal[1] = 1
    goal: str = Field(min_length=1)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    results: list[ToolResult] = Field(default_factory=list)
    approvals: list[ApprovalRecord] = Field(default_factory=list)
    status: Literal["completed", "halted"]
    cluster_execution: Literal["disabled", "controlled"] = "disabled"
    human_review_required: Literal[True] = True

    @model_validator(mode="after")
    def results_align_with_approvals(self) -> AgentTranscript:
        result_ids = [result.call_id for result in self.results]
        approval_ids = [record.call_id for record in self.approvals]
        if result_ids != approval_ids:
            raise ValueError("transcript results and approvals must align by call id")
        for result, approval in zip(self.results, self.approvals, strict=True):
            if result.risk_tier != RiskTier.CLUSTER_EXECUTE:
                continue
            if (
                result.tool not in CONTROLLED_CLUSTER_EXECUTE_TOOLS
                or approval.tool != result.tool
            ):
                raise ValueError(
                    "only fixed controlled tools may produce cluster_execute results"
                )
            if approval.risk_tier != RiskTier.CLUSTER_EXECUTE:
                raise ValueError("cluster_execute result risk tier does not match approval")
            if result.status == "blocked":
                if result.approval_state != approval.state:
                    raise ValueError("blocked cluster_execute result must match approval state")
                continue
            if approval.state != ApprovalState.APPROVED or not (
                approval.approver and approval.approver.strip()
            ):
                raise ValueError(
                    "controlled cluster result requires explicit named approval"
                )
            if result.status == "ok" and result.approval_state != ApprovalState.EXECUTED:
                raise ValueError(
                    "successful controlled cluster result must be marked executed"
                )

        expected_cluster_execution = transcript_cluster_execution(
            self.results, self.approvals
        )
        if self.cluster_execution != expected_cluster_execution:
            raise ValueError(
                "cluster_execution must be controlled only for an explicitly "
                "approved successful controlled cluster result"
            )
        return self


def transcript_cluster_execution(
    results: list[ToolResult], approvals: list[ApprovalRecord]
) -> Literal["disabled", "controlled"]:
    """Summarize whether this transcript contains a controlled cluster action."""

    approvals_by_id = {record.call_id: record for record in approvals}
    for result in results:
        approval = approvals_by_id.get(result.call_id)
        if (
            result.tool in CONTROLLED_CLUSTER_EXECUTE_TOOLS
            and result.risk_tier == RiskTier.CLUSTER_EXECUTE
            and result.status == "ok"
            and result.approval_state == ApprovalState.EXECUTED
            and approval is not None
            and approval.tool == result.tool
            and approval.risk_tier == RiskTier.CLUSTER_EXECUTE
            and approval.state == ApprovalState.APPROVED
            and approval.approver is not None
            and bool(approval.approver.strip())
        ):
            return "controlled"
    return "disabled"
