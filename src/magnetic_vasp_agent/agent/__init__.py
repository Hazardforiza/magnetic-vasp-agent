"""Framework-agnostic agent tool protocol, approval gate, and runner.

This package exposes the deterministic VASP core as typed, auditable tools an
LLM agent may call. The protocol, approval state machine, and runner have no
dependency on any LLM SDK or network, so they can be tested fully offline. The
optional LangGraph binding lives in :mod:`magnetic_vasp_agent.agent.graph` and is
imported only when ``langgraph``/``langchain`` are installed.

Hard boundaries carried over from the rest of the project:

* An agent can only invoke tools that exist in the registry; free-form shell,
  SSH, or MPI strings are never accepted.
* Every typed request still passes the existing deterministic validators.
* Cluster execution stays disabled: no launch/cancel/retry tool is registered
  and the ``cluster_execute`` risk tier is always blocked.
* POTCAR content is never placed in a tool result or sent to an LLM.
"""

from magnetic_vasp_agent.agent.approval import ApprovalPolicy, decide_approval
from magnetic_vasp_agent.agent.runner import run_plan
from magnetic_vasp_agent.agent.tool_models import (
    AgentPlan,
    AgentTranscript,
    ApprovalRecord,
    ApprovalState,
    RiskTier,
    ToolCall,
    ToolResult,
    ToolSpec,
    sha256_of,
)
from magnetic_vasp_agent.agent.tools import ToolRegistry, default_registry

__all__ = [
    "AgentPlan",
    "AgentTranscript",
    "ApprovalPolicy",
    "ApprovalRecord",
    "ApprovalState",
    "RiskTier",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "decide_approval",
    "default_registry",
    "run_plan",
    "sha256_of",
]
