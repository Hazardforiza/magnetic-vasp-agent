"""Tiered approval state machine for the local VASP agent.

The gate decides, per tool call, whether execution may proceed. It never
inspects free-form strings; it keys purely off the tool's declared
:class:`RiskTier`, so an LLM cannot argue its way past a boundary.

Policy summary:

* ``read_only``      -> auto-approved.
* ``local_write``    -> allowed when ``allow_local_writes`` is set (default),
  and always recorded.
* ``cluster_plan``   -> requires the call id to appear in
  ``approved_cluster_plan_call_ids`` (an explicit human approval); otherwise the
  run halts awaiting approval.
* ``cluster_execute``-> only fixed smoke, transfer, and launch calls may proceed,
  and only after each call id has a named human approval.
"""

from __future__ import annotations

from pydantic import Field

from magnetic_vasp_agent.agent.tool_models import (
    CONTROLLED_CLUSTER_EXECUTE_TOOLS,
    ApprovalRecord,
    ApprovalState,
    RiskTier,
    StrictModel,
    ToolCall,
    ToolSpec,
)


class ApprovalPolicy(StrictModel):
    """Human-controlled policy; the agent cannot modify it."""

    allow_local_writes: bool = True
    approved_cluster_plan_call_ids: list[str] = Field(default_factory=list)
    approved_cluster_execute_call_ids: list[str] = Field(default_factory=list)
    approver: str | None = None


def decide_approval(
    call: ToolCall, spec: ToolSpec, policy: ApprovalPolicy
) -> ApprovalRecord:
    """Return the approval outcome for one validated tool call."""

    tier = spec.risk_tier
    if tier == RiskTier.READ_ONLY:
        return ApprovalRecord(
            call_id=call.call_id,
            tool=call.tool,
            risk_tier=tier,
            state=ApprovalState.APPROVED,
            reason="read-only tools are auto-approved",
        )
    if tier == RiskTier.LOCAL_WRITE:
        if policy.allow_local_writes:
            return ApprovalRecord(
                call_id=call.call_id,
                tool=call.tool,
                risk_tier=tier,
                state=ApprovalState.APPROVED,
                reason="local write permitted by policy",
                approver=policy.approver,
            )
        return ApprovalRecord(
            call_id=call.call_id,
            tool=call.tool,
            risk_tier=tier,
            state=ApprovalState.BLOCKED,
            reason="local writes disabled by policy",
        )
    if tier == RiskTier.CLUSTER_PLAN:
        if call.call_id in policy.approved_cluster_plan_call_ids:
            if not (policy.approver and policy.approver.strip()):
                return ApprovalRecord(
                    call_id=call.call_id,
                    tool=call.tool,
                    risk_tier=tier,
                    state=ApprovalState.BLOCKED,
                    reason="cluster plan approval requires a named approver",
                )
            return ApprovalRecord(
                call_id=call.call_id,
                tool=call.tool,
                risk_tier=tier,
                state=ApprovalState.APPROVED,
                reason="cluster plan explicitly approved by human",
                approver=policy.approver,
            )
        return ApprovalRecord(
            call_id=call.call_id,
            tool=call.tool,
            risk_tier=tier,
            state=ApprovalState.AWAITING_APPROVAL,
            reason="cluster plan requires explicit human approval",
        )
    if tier == RiskTier.CLUSTER_EXECUTE:
        if (
            call.tool not in CONTROLLED_CLUSTER_EXECUTE_TOOLS
            or spec.name != call.tool
        ):
            return ApprovalRecord(
                call_id=call.call_id,
                tool=call.tool,
                risk_tier=tier,
                state=ApprovalState.BLOCKED,
                reason="only fixed controlled tools may execute on cluster",
            )
        if call.call_id in policy.approved_cluster_execute_call_ids:
            if not (policy.approver and policy.approver.strip()):
                return ApprovalRecord(
                    call_id=call.call_id,
                    tool=call.tool,
                    risk_tier=tier,
                    state=ApprovalState.BLOCKED,
                    reason="cluster execution approval requires a named approver",
                )
            return ApprovalRecord(
                call_id=call.call_id,
                tool=call.tool,
                risk_tier=tier,
                state=ApprovalState.APPROVED,
                reason="cluster execution explicitly approved by human",
                approver=policy.approver,
            )
        return ApprovalRecord(
            call_id=call.call_id,
            tool=call.tool,
            risk_tier=tier,
            state=ApprovalState.AWAITING_APPROVAL,
            reason="cluster execution requires explicit human approval",
        )
    # unknown/elevated tier.
    return ApprovalRecord(
        call_id=call.call_id,
        tool=call.tool,
        risk_tier=tier,
        state=ApprovalState.BLOCKED,
        reason="cluster execution is disabled in this build",
    )
