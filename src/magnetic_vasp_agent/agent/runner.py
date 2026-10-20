"""Deterministic executor for an :class:`AgentPlan`.

The runner walks the plan call by call: it looks each tool up in the registry,
validates the typed arguments, asks the approval gate for a decision, and only
then runs the deterministic function. It never calls an LLM. Registered cluster
execution is limited to fixed, separately approved smoke, transfer, and launch
tools. On the first non-success outcome it halts, because later steps usually
depend on earlier ones.
"""

from __future__ import annotations

from pydantic import ValidationError

from magnetic_vasp_agent.agent.approval import ApprovalPolicy, decide_approval
from magnetic_vasp_agent.agent.tool_models import (
    AgentPlan,
    AgentTranscript,
    ApprovalRecord,
    ApprovalState,
    RiskTier,
    ToolCall,
    ToolResult,
    sha256_of,
    transcript_cluster_execution,
)
from magnetic_vasp_agent.agent.tools import Tool, ToolRegistry, default_registry


def _unknown_tool_records(call: ToolCall) -> tuple[ApprovalRecord, ToolResult]:
    approval = ApprovalRecord(
        call_id=call.call_id,
        tool=call.tool,
        risk_tier=RiskTier.READ_ONLY,
        state=ApprovalState.BLOCKED,
        reason="unknown tool; not in the registry",
    )
    result = ToolResult(
        call_id=call.call_id,
        tool=call.tool,
        risk_tier=RiskTier.READ_ONLY,
        approval_state=ApprovalState.BLOCKED,
        status="blocked",
        message=f"unknown tool {call.tool!r}",
    )
    return approval, result


def _execute(call: ToolCall, tool: Tool) -> ToolResult:
    """Validate arguments, run the tool, and map failures to typed statuses."""

    try:
        data = tool.input_model.model_validate(call.arguments)
    except ValidationError as exc:
        return ToolResult(
            call_id=call.call_id,
            tool=call.tool,
            risk_tier=tool.spec.risk_tier,
            approval_state=ApprovalState.REJECTED,
            status="validation_error",
            message=f"argument validation failed: {exc.error_count()} error(s)",
        )
    try:
        output = tool.run(data)
    except ValueError as exc:
        return ToolResult(
            call_id=call.call_id,
            tool=call.tool,
            risk_tier=tool.spec.risk_tier,
            approval_state=ApprovalState.APPROVED,
            status="validation_error",
            message=str(exc),
        )
    except (OSError, KeyError, TypeError, RuntimeError) as exc:
        return ToolResult(
            call_id=call.call_id,
            tool=call.tool,
            risk_tier=tool.spec.risk_tier,
            approval_state=ApprovalState.APPROVED,
            status="error",
            message=str(exc),
        )
    return ToolResult(
        call_id=call.call_id,
        tool=call.tool,
        risk_tier=tool.spec.risk_tier,
        approval_state=ApprovalState.EXECUTED,
        status="ok",
        output=output,
        output_sha256=sha256_of(output),
    )


def evaluate_call(
    call: ToolCall,
    *,
    policy: ApprovalPolicy | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[ApprovalRecord, ToolResult]:
    """Gate and run one call, returning its approval record and result.

    Shared by the offline :func:`run_plan` and the LangGraph binding so both
    apply the exact same registry lookup, argument validation, and approval policy.
    """

    policy = policy or ApprovalPolicy()
    registry = registry or default_registry()

    if call.tool not in registry:
        return _unknown_tool_records(call)

    tool = registry.get(call.tool)
    decision = decide_approval(call, tool.spec, policy)
    if decision.state != ApprovalState.APPROVED:
        result = ToolResult(
            call_id=call.call_id,
            tool=call.tool,
            risk_tier=tool.spec.risk_tier,
            approval_state=decision.state,
            status="blocked",
            message=decision.reason,
        )
        return decision, result

    if call.tool == "execution.resource_grant":
        requested_approver = call.arguments.get("approver")
        approved_by = decision.approver
        if not (
            isinstance(requested_approver, str)
            and approved_by is not None
            and requested_approver.strip() == approved_by.strip()
        ):
            blocked = ApprovalRecord(
                call_id=call.call_id,
                tool=call.tool,
                risk_tier=tool.spec.risk_tier,
                state=ApprovalState.BLOCKED,
                reason=(
                    "resource grant approver argument must match the named "
                    "human who approved this cluster_plan call"
                ),
                approver=approved_by,
            )
            return blocked, ToolResult(
                call_id=call.call_id,
                tool=call.tool,
                risk_tier=tool.spec.risk_tier,
                approval_state=ApprovalState.BLOCKED,
                status="blocked",
                message=blocked.reason,
            )

    return decision, _execute(call, tool)


def run_plan(
    plan: AgentPlan,
    *,
    policy: ApprovalPolicy | None = None,
    registry: ToolRegistry | None = None,
) -> AgentTranscript:
    """Run a typed plan through the same approval gates used by LangGraph."""

    policy = policy or ApprovalPolicy()
    registry = registry or default_registry()
    approvals: list[ApprovalRecord] = []
    results: list[ToolResult] = []
    status = "completed"

    for call in plan.calls:
        approval, result = evaluate_call(call, policy=policy, registry=registry)
        approvals.append(approval)
        results.append(result)
        if result.status != "ok":
            status = "halted"
            break

    return AgentTranscript(
        goal=plan.goal,
        plan_sha256=plan.digest(),
        results=results,
        approvals=approvals,
        status=status,
        cluster_execution=transcript_cluster_execution(results, approvals),
    )
