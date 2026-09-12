"""Optional LangGraph binding for the local VASP agent.

This module is a thin adapter over the framework-agnostic core. Every tool call
still routes through :func:`evaluate_call`, so the same registry lookup, argument
validation, approval gate, and provenance apply whether a human or the LLM
proposes a call. The LLM can only invoke registered tools; there is no shell,
SSH, or MPI tool, and cluster execution stays disabled.

Why LangGraph: the approval state machine maps directly onto a graph, and the
``cluster_plan`` "wait for a human" step is expressed with LangGraph's native
``interrupt`` primitive, giving real human-in-the-loop without loosening any
boundary.

``langgraph`` / ``langchain`` are imported lazily so the rest of the package
(and its tests) work without them installed.
"""

from __future__ import annotations

import operator
import re
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict

from pydantic import Field

from magnetic_vasp_agent.agent.approval import ApprovalPolicy
from magnetic_vasp_agent.agent.llm_config import LlmConfig
from magnetic_vasp_agent.agent.runner import evaluate_call
from magnetic_vasp_agent.agent.tool_models import (
    AgentTranscript,
    ApprovalRecord,
    RiskTier,
    StrictModel,
    ToolCall,
    ToolResult,
    sha256_of,
)
from magnetic_vasp_agent.agent.tools import ToolRegistry, default_registry

_PLANNER_INSTRUCTIONS = (
    "You plan magnetic VASP calculations by calling the provided tools only. "
    "Never invent shell, SSH, or MPI commands. Every tool result is advisory and "
    "requires human review; cluster execution is disabled. If a tool is blocked "
    "or returns an error, stop and report it rather than working around it. When "
    "you have enough information, reply with a short plain-text summary and no "
    "further tool calls."
)


class PendingApproval(StrictModel):
    """A cluster_plan call paused for explicit human approval."""

    call_id: str
    tool: str
    risk_tier: RiskTier
    arguments: dict[str, Any]
    reason: str


class GraphRunResult(StrictModel):
    """What the LangGraph path returns: an auditable transcript plus context."""

    goal: str = Field(min_length=1)
    transcript: AgentTranscript
    llm_output: str
    pending_approvals: list[PendingApproval] = Field(default_factory=list)
    thread_id: str


def langgraph_available() -> bool:
    try:
        import langchain_core  # noqa: F401
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


def _require_langgraph() -> None:
    if not langgraph_available():
        raise RuntimeError(
            "langgraph is not installed; install the optional 'agent' extra to use "
            "the LLM path (pip install 'magnetic-vasp-agent[agent]')"
        )


def build_chat_model(config: LlmConfig):
    """Build a LangChain chat model for an OpenAI-compatible endpoint."""

    _require_langgraph()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "langchain-openai is required for the LLM path; install the 'agent' extra"
        ) from exc

    # The config model id follows the litellm style ("openai/<model>"); the
    # OpenAI-compatible client wants the bare model id.
    model_id = config.model.split("/", 1)[1] if config.model.startswith("openai/") else config.model
    return ChatOpenAI(
        model=model_id,
        base_url=config.base_url,
        api_key=config.resolve_api_key(),
        temperature=config.temperature,
        timeout=config.timeout_seconds,
    )


def _openai_tools(registry: ToolRegistry) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return OpenAI-format tool schemas and a sanitized-name -> tool-name map.

    Registered tool names use dots (``incar.build``); OpenAI function names may
    not, so we expose an underscore form and map it back on the way in.
    """

    schemas: list[dict[str, Any]] = []
    name_map: dict[str, str] = {}
    for name in registry.names():
        spec = registry.get(name).spec
        sanitized = name.replace(".", "_")
        name_map[sanitized] = name
        name_map[name] = name  # tolerate a model that echoes the dotted name
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": sanitized,
                    "description": spec.description,
                    "parameters": registry.get(name).input_model.model_json_schema(),
                },
            }
        )
    return schemas, name_map


def build_graph(
    model,
    *,
    registry: ToolRegistry | None = None,
    policy: ApprovalPolicy | None = None,
):
    """Compile the planner/tools graph. ``model`` must accept a message list.

    ``model`` may be a real bound chat model or a fake one (for offline tests);
    the graph never touches the network itself.
    """

    _require_langgraph()
    from langchain_core.messages import ToolMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.types import interrupt

    registry = registry or default_registry()
    policy = policy or ApprovalPolicy()
    _, name_map = _openai_tools(registry)

    # Functional TypedDict form: values are real objects, not stringized
    # annotations, so ``from __future__ import annotations`` does not hide the
    # locally imported ``add_messages`` reducer from LangGraph.
    _State = TypedDict(
        "_State",
        {
            "messages": Annotated[list, add_messages],
            "goal": str,
            "approvals": Annotated[list, operator.add],
            "results": Annotated[list, operator.add],
            "call_seq": int,
            "policy": dict,
        },
    )

    def planner(state):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def route(state):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def tools(state):
        last = state["messages"][-1]
        base_policy = ApprovalPolicy.model_validate(state["policy"])
        seq = state["call_seq"]

        calls: list[tuple[str, ToolCall]] = []  # (tool_call_id, ToolCall)
        for index, tool_call in enumerate(last.tool_calls):
            call_id = f"T{seq + index + 1}"
            registry_name = name_map.get(tool_call["name"], tool_call["name"])
            calls.append(
                (
                    tool_call.get("id") or call_id,
                    ToolCall(
                        call_id=call_id,
                        tool=registry_name,
                        arguments=tool_call.get("args", {}) or {},
                    ),
                )
            )

        # Collect every cluster_plan approval up front, before any side effect,
        # so a resumed node never re-executes an earlier tool.
        needs_approval: list[PendingApproval] = []
        for _tc_id, call in calls:
            if call.tool not in registry:
                continue
            spec = registry.get(call.tool).spec
            if (
                spec.risk_tier == RiskTier.CLUSTER_PLAN
                and call.call_id not in base_policy.approved_cluster_plan_call_ids
            ):
                needs_approval.append(
                    PendingApproval(
                        call_id=call.call_id,
                        tool=call.tool,
                        risk_tier=spec.risk_tier,
                        arguments=call.arguments,
                        reason="cluster plan requires explicit human approval",
                    )
                )

        working_policy = base_policy
        if needs_approval:
            decisions = interrupt(
                {"pending_approvals": [item.model_dump(mode="json") for item in needs_approval]}
            )
            decisions = decisions or {}
            approved_ids = list(base_policy.approved_cluster_plan_call_ids)
            approver = base_policy.approver
            for item in needs_approval:
                decision = decisions.get(item.call_id, {}) if isinstance(decisions, dict) else {}
                if decision.get("approve"):
                    approved_ids.append(item.call_id)
                    approver = decision.get("approver") or approver
            working_policy = base_policy.model_copy(
                update={"approved_cluster_plan_call_ids": approved_ids, "approver": approver}
            )

        tool_messages = []
        approvals: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for tc_id, call in calls:
            approval, result = evaluate_call(
                call, policy=working_policy, registry=registry
            )
            approvals.append(approval.model_dump(mode="json"))
            results.append(result.model_dump(mode="json"))
            tool_messages.append(
                ToolMessage(
                    content=result.model_dump_json(),
                    tool_call_id=tc_id,
                    name=call.tool,
                )
            )

        return {
            "messages": tool_messages,
            "approvals": approvals,
            "results": results,
            "call_seq": seq + len(calls),
        }

    builder = StateGraph(_State)
    builder.add_node("planner", planner)
    builder.add_node("tools", tools)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "planner")
    return builder.compile(checkpointer=MemorySaver())


def _build_transcript(goal: str, state: dict[str, Any]) -> AgentTranscript:
    # These dicts were produced by our own model_dump(mode="json"); rebuild them
    # with strict=False so JSON string enums coerce back to their members.
    approvals = [
        ApprovalRecord.model_validate(item, strict=False)
        for item in state.get("approvals", [])
    ]
    results = [
        ToolResult.model_validate(item, strict=False)
        for item in state.get("results", [])
    ]
    plan_sha256 = sha256_of([{"call_id": r.call_id, "tool": r.tool} for r in results])
    status = "completed" if results and all(r.status == "ok" for r in results) else "halted"
    return AgentTranscript(
        goal=goal,
        plan_sha256=plan_sha256,
        results=results,
        approvals=approvals,
        status=status,
    )


_SENTINEL_RE = re.compile(r"\s*(?:<\|[^>]*\|>|<[A-Z0-9_]+>)\s*$")


def _strip_sentinels(text: str) -> str:
    """Drop trailing end-of-turn sentinels some proxies/models append.

    Only removes a sentinel token anchored at the very end (e.g. ``<CPA_DONE>``
    or ``<|end|>``); ordinary prose and mid-text angle brackets are untouched.
    """

    previous = None
    while previous != text:
        previous = text
        text = _SENTINEL_RE.sub("", text)
    return text.rstrip()


def _last_ai_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        content = getattr(message, "content", None)
        if content and not getattr(message, "tool_calls", None):
            text = content if isinstance(content, str) else str(content)
            return _strip_sentinels(text)
    return ""


class VaspAgentGraph:
    """A compiled planner/tools graph with a persistent checkpointer.

    Holding one instance lets a paused ``cluster_plan`` approval be resumed on
    the same in-memory checkpoint, which is the point of the LangGraph binding.
    """

    def __init__(
        self,
        config: LlmConfig,
        *,
        policy: ApprovalPolicy | None = None,
        registry: ToolRegistry | None = None,
        model: Any | None = None,
    ) -> None:
        _require_langgraph()
        self._config = config
        self._policy = policy or ApprovalPolicy()
        self._registry = registry or default_registry()
        if model is None:
            schemas, _ = _openai_tools(self._registry)
            model = build_chat_model(config).bind_tools(schemas)
        self._graph = build_graph(model, registry=self._registry, policy=self._policy)

    def _run_config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._config.max_tool_calls * 2 + 5,
        }

    def run(self, goal: str, *, thread_id: str | None = None) -> GraphRunResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        thread_id = thread_id or str(uuid.uuid4())
        initial: dict[str, Any] = {
            "messages": [
                SystemMessage(content=_PLANNER_INSTRUCTIONS),
                HumanMessage(content=f"{goal}\n\n{_PLANNER_INSTRUCTIONS}"),
            ],
            "goal": goal,
            "approvals": [],
            "results": [],
            "call_seq": 0,
            "policy": self._policy.model_dump(mode="json"),
        }
        output = self._graph.invoke(initial, self._run_config(thread_id))
        return self._collect(goal, thread_id, output)

    def resume(
        self, result: GraphRunResult, decisions: dict[str, dict[str, Any]]
    ) -> GraphRunResult:
        """Resume a paused run with per-call human approval decisions.

        ``decisions`` maps a call id to ``{"approve": bool, "approver": str}``.
        The agent can never approve its own cluster_plan calls.
        """

        from langgraph.types import Command

        output = self._graph.invoke(
            Command(resume=decisions), self._run_config(result.thread_id)
        )
        return self._collect(result.goal, result.thread_id, output)

    def _collect(self, goal: str, thread_id: str, output: Any) -> GraphRunResult:
        pending: list[PendingApproval] = []
        interrupts = output.get("__interrupt__") if isinstance(output, dict) else None
        if interrupts:
            for item in interrupts:
                value = getattr(item, "value", item)
                for entry in (value or {}).get("pending_approvals", []):
                    pending.append(PendingApproval.model_validate(entry, strict=False))

        snapshot = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        state = snapshot.values if snapshot else output
        transcript = _build_transcript(goal, state)
        return GraphRunResult(
            goal=goal,
            transcript=transcript,
            llm_output=_last_ai_text(state),
            pending_approvals=pending,
            thread_id=thread_id,
        )


def run_goal(
    goal: str,
    config: LlmConfig,
    *,
    policy: ApprovalPolicy | None = None,
    registry: ToolRegistry | None = None,
    model: Any | None = None,
) -> GraphRunResult:
    """Turn a natural-language goal into gated tool calls via LangGraph.

    ``model`` may be supplied directly (for example a fake chat model in tests)
    to avoid any network access; otherwise it is built from ``config`` and bound
    to the registered tool schemas. For resumable cluster_plan approval use
    :class:`VaspAgentGraph` directly so the checkpoint persists.
    """

    _require_langgraph()
    return VaspAgentGraph(
        config, policy=policy, registry=registry, model=model
    ).run(goal)


def run_goal_from_config_path(
    goal: str,
    config_path: Path,
    *,
    policy: ApprovalPolicy | None = None,
) -> GraphRunResult:
    from magnetic_vasp_agent.agent.llm_config import load_llm_config

    return run_goal(goal, load_llm_config(config_path), policy=policy)
