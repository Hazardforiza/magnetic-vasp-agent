from __future__ import annotations

from magnetic_vasp_agent.agent.approval import ApprovalPolicy, decide_approval
from magnetic_vasp_agent.agent.tool_models import (
    ApprovalState,
    RiskTier,
    ToolCall,
    ToolSpec,
)


def _spec(tier: RiskTier) -> ToolSpec:
    return ToolSpec(
        name="tool.x",
        description="a tool",
        risk_tier=tier,
        produces_files=tier != RiskTier.READ_ONLY,
        input_schema={"type": "object"},
    )


def _call() -> ToolCall:
    return ToolCall(call_id="T1", tool="tool.x", arguments={})


def test_read_only_is_auto_approved() -> None:
    record = decide_approval(_call(), _spec(RiskTier.READ_ONLY), ApprovalPolicy())
    assert record.state == ApprovalState.APPROVED


def test_local_write_allowed_by_default() -> None:
    record = decide_approval(_call(), _spec(RiskTier.LOCAL_WRITE), ApprovalPolicy())
    assert record.state == ApprovalState.APPROVED


def test_local_write_can_be_disabled() -> None:
    policy = ApprovalPolicy(allow_local_writes=False)
    record = decide_approval(_call(), _spec(RiskTier.LOCAL_WRITE), policy)
    assert record.state == ApprovalState.BLOCKED


def test_cluster_plan_without_approval_awaits() -> None:
    record = decide_approval(_call(), _spec(RiskTier.CLUSTER_PLAN), ApprovalPolicy())
    assert record.state == ApprovalState.AWAITING_APPROVAL


def test_cluster_plan_approved_id_still_needs_named_approver() -> None:
    policy = ApprovalPolicy(approved_cluster_plan_call_ids=["T1"])
    record = decide_approval(_call(), _spec(RiskTier.CLUSTER_PLAN), policy)
    assert record.state == ApprovalState.BLOCKED


def test_cluster_plan_approved_with_named_approver() -> None:
    policy = ApprovalPolicy(
        approved_cluster_plan_call_ids=["T1"], approver="researcher"
    )
    record = decide_approval(_call(), _spec(RiskTier.CLUSTER_PLAN), policy)
    assert record.state == ApprovalState.APPROVED
    assert record.approver == "researcher"


def test_cluster_execute_is_always_blocked() -> None:
    policy = ApprovalPolicy(
        approved_cluster_plan_call_ids=["T1"], approver="researcher"
    )
    record = decide_approval(_call(), _spec(RiskTier.CLUSTER_EXECUTE), policy)
    assert record.state == ApprovalState.BLOCKED
