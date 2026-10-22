from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from magnetic_vasp_agent.agent.graph import (
    _content_to_text,
    _strip_sentinels,
    langgraph_available,
    run_goal,
)
from magnetic_vasp_agent.agent.llm_config import LlmConfig, load_llm_config
from magnetic_vasp_agent.agent.tool_models import ApprovalState, RiskTier

FIXTURES = Path(__file__).parent / "fixtures"
MN3AL_POSCAR = str(FIXTURES / "Mn3Al" / "POSCAR")


def _config() -> LlmConfig:
    return LlmConfig(
        model="openai/test-model",
        base_url="https://gateway.example.com/v1",
        api_key_env="MAGVASP_TEST_KEY",
    )


def test_content_to_text_reads_responses_blocks() -> None:
    assert _content_to_text("plain") == "plain"
    assert (
        _content_to_text(
            [
                {"type": "reasoning", "summary": []},
                {"type": "text", "text": "结构检查完成。"},
            ]
        )
        == "结构检查完成。"
    )


def test_strip_sentinels_removes_trailing_markers_only() -> None:
    assert _strip_sentinels("mesh summary <CPA_DONE>") == "mesh summary"
    assert _strip_sentinels("done <|end|>") == "done"
    assert _strip_sentinels("done <CPA_DONE> <|end|>") == "done"
    # Ordinary prose and mid-text angle brackets are untouched.
    assert _strip_sentinels("use a < b comparison") == "use a < b comparison"
    assert _strip_sentinels("value <T1> is used here") == "value <T1> is used here"


# --------------------------------------------------------------------------- #
# LLM config: local key is masked; environment variable remains supported.     #
# --------------------------------------------------------------------------- #


def test_resolve_api_key_reads_inline_and_masks_it(monkeypatch) -> None:
    secret = "sk-inline-secret"
    monkeypatch.setenv("MAGVASP_TEST_KEY", "sk-from-env")
    config = LlmConfig(
        model="openai/test-model",
        base_url="https://gateway.example.com/v1",
        api_key_env="MAGVASP_TEST_KEY",
        api_key=secret,
    )
    assert config.resolve_api_key() == secret
    assert secret not in repr(config)
    assert secret not in config.model_dump_json()


def test_resolve_api_key_falls_back_to_environment(monkeypatch) -> None:
    monkeypatch.setenv("MAGVASP_TEST_KEY", "sk-from-env")
    config = _config()
    assert config.resolve_api_key() == "sk-from-env"


def test_config_rejects_api_key_as_env_name_without_echoing_it() -> None:
    secret = "sk-live-secret-value"
    with pytest.raises(ValidationError) as exc_info:
        LlmConfig(
            model="openai/test-model",
            base_url="https://gateway.example.com/v1",
            api_key_env=secret,
        )
    assert secret not in str(exc_info.value)


def test_config_rejects_non_http_base_url() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(model="openai/x", base_url="ftp://host/v1")


def test_resolve_api_key_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("MAGVASP_TEST_KEY", "sk-from-env")
    assert _config().resolve_api_key() == "sk-from-env"


def test_resolve_api_key_missing_raises(monkeypatch) -> None:
    monkeypatch.delenv("MAGVASP_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="environment variable named"):
        _config().resolve_api_key()


def test_config_accepts_responses_wire_and_reasoning() -> None:
    config = LlmConfig(
        model="gpt-5.6-sol",
        base_url="https://wawazz.xyz",
        wire_api="responses",
        reasoning_effort="high",
        disable_response_storage=True,
    )
    assert config.wire_api == "responses"
    assert config.reasoning_effort == "high"
    assert config.disable_response_storage is True


def test_example_config_loads(tmp_path) -> None:
    path = tmp_path / "llm.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "openai_compatible",
                "model": "openai/deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
                "api_key_env": "MAGVASP_LLM_API_KEY",
            }
        ),
        encoding="utf-8",
    )
    assert load_llm_config(path).model == "openai/deepseek-chat"


# --------------------------------------------------------------------------- #
# LangGraph binding.                                                            #
# --------------------------------------------------------------------------- #


def test_run_goal_requires_langgraph_when_missing() -> None:
    if langgraph_available():
        pytest.skip("langgraph is installed; missing-dependency path not exercised")
    with pytest.raises(RuntimeError, match="langgraph is not installed"):
        run_goal("plan a static SCF for Mn3Al", _config())


def _scripted_model(responses):
    """A LangChain chat model that replays scripted AIMessages, no network."""

    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    class ScriptedChatModel(BaseChatModel):
        scripted: list
        _cursor: int = PrivateAttr(default=0)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            index = min(self._cursor, len(self.scripted) - 1)
            self._cursor += 1
            return ChatResult(
                generations=[ChatGeneration(message=self.scripted[index])]
            )

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

    return ScriptedChatModel(scripted=responses)


@pytest.mark.skipif(not langgraph_available(), reason="langgraph not installed")
def test_read_only_tool_call_runs_through_gate() -> None:
    from langchain_core.messages import AIMessage

    model = _scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "structure_show",
                        "args": {"poscar": MN3AL_POSCAR},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Inspected the Mn3Al structure; review required."),
        ]
    )
    result = run_goal("inspect Mn3Al", _config(), model=model)

    assert result.transcript.cluster_execution == "disabled"
    assert result.transcript.human_review_required is True
    assert result.pending_approvals == []
    assert len(result.transcript.results) == 1
    only = result.transcript.results[0]
    assert only.tool == "structure.show"
    assert only.status == "ok"
    assert only.approval_state == ApprovalState.EXECUTED


@pytest.mark.skipif(not langgraph_available(), reason="langgraph not installed")
def test_cluster_plan_pauses_then_resumes_with_human_approval() -> None:
    from langchain_core.messages import AIMessage

    from magnetic_vasp_agent.agent.graph import VaspAgentGraph

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execution_transfer_plan",
                    "args": {"bundle": "b", "profile": "p", "request": "r"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="Transfer plan attempted; review required."),
    ]
    agent = VaspAgentGraph(_config(), model=_scripted_model(responses))

    paused = agent.run("upload the bundle")
    # The cluster_plan call halts for a human; nothing executed yet.
    assert paused.pending_approvals, "cluster_plan should pause for approval"
    assert paused.pending_approvals[0].tool == "execution.transfer_plan"
    assert paused.transcript.results == []

    resumed = agent.resume(
        paused,
        {paused.pending_approvals[0].call_id: {"approve": True, "approver": "researcher"}},
    )
    # After approval the gate no longer blocks; the tool now fails on missing
    # files rather than being refused, proving approval succeeded.
    assert resumed.pending_approvals == []
    assert resumed.transcript.results
    first = resumed.transcript.results[0]
    assert first.tool == "execution.transfer_plan"
    assert first.status != "blocked"


@pytest.mark.skipif(not langgraph_available(), reason="langgraph not installed")
def test_cluster_execute_pauses_then_resumes_with_named_approval() -> None:
    from langchain_core.messages import AIMessage

    from magnetic_vasp_agent.agent.graph import VaspAgentGraph
    from magnetic_vasp_agent.agent.tools import StructureShowInput, ToolRegistry

    registry = ToolRegistry()
    registry.register(
        name="execution.launch",
        description="offline launch stub",
        risk_tier=RiskTier.CLUSTER_EXECUTE,
        produces_files=True,
        input_model=StructureShowInput,
        run=lambda data: {"state": "running", "offline_stub": True},
    )
    model = _scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execution_launch",
                        "args": {"poscar": MN3AL_POSCAR},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Controlled launch call completed; review required."),
        ]
    )
    agent = VaspAgentGraph(_config(), model=model, registry=registry)

    paused = agent.run("exercise launch approval offline")
    assert len(paused.pending_approvals) == 1
    assert paused.pending_approvals[0].tool == "execution.launch"
    assert paused.pending_approvals[0].risk_tier == RiskTier.CLUSTER_EXECUTE
    assert paused.transcript.results == []

    resumed = agent.resume(
        paused,
        {paused.pending_approvals[0].call_id: {"approve": True, "approver": "researcher"}},
    )
    assert resumed.pending_approvals == []
    assert resumed.transcript.status == "completed"
    assert resumed.transcript.cluster_execution == "controlled"
    assert resumed.transcript.results[0].tool == "execution.launch"
    assert resumed.transcript.results[0].status == "ok"


@pytest.mark.skipif(not langgraph_available(), reason="langgraph not installed")
def test_no_tool_calls_yields_empty_halted_transcript() -> None:
    from langchain_core.messages import AIMessage

    model = _scripted_model([AIMessage(content="No tools needed; here is advice.")])
    result = run_goal("just answer", _config(), model=model)
    assert result.transcript.status == "halted"
    assert result.transcript.results == []
    assert result.transcript.cluster_execution == "disabled"
