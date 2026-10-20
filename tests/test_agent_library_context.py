"""Exercise the web-to-Agent retrieval path with a scripted model, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from magnetic_vasp_agent.agent.approval import ApprovalPolicy
from magnetic_vasp_agent.agent.graph import langgraph_available
from magnetic_vasp_agent.agent.library_context import register_library_search
from magnetic_vasp_agent.agent.runner import evaluate_call
from magnetic_vasp_agent.agent.tool_models import ToolCall
from magnetic_vasp_agent.agent.tools import ToolRegistry
from magnetic_vasp_agent.web import create_app


def test_bound_library_tool_rejects_paths_and_mutations() -> None:
    registry = ToolRegistry()
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return {"query": kwargs["query"], "evidence": [], "warnings": []}

    register_library_search(registry, search)
    for arguments in (
        {"query": "Wannier", "index": "C:/private"},
        {"query": "Wannier", "scope": "../../private"},
        {"query": "Wannier", "limit": 1000},
    ):
        result = evaluate_call(
            ToolCall(call_id="T1", tool="knowledge.library_search", arguments=arguments),
            registry=registry, policy=ApprovalPolicy(),
        )[1]
        assert result.status == "validation_error"
    assert not calls
    assert registry.names() == ["knowledge.library_search"]
    result = evaluate_call(
        ToolCall(call_id="T2", tool="knowledge.library_search", arguments={"query": "Wannier"}),
        registry=registry, policy=ApprovalPolicy(),
    )[1]
    assert result.status == "ok"
    assert calls == [{"query": "Wannier", "scope": "all", "limit": 5}]


@pytest.mark.skipif(not langgraph_available(), reason="agent extra not installed")
def test_confirmed_web_source_reaches_agent_tool_context(tmp_path: Path, monkeypatch) -> None:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    from magnetic_vasp_agent.agent.llm_config import LlmConfig

    observed = []

    class LibraryModel(BaseChatModel):
        _cursor: int = PrivateAttr(default=0)

        @property
        def _llm_type(self):
            return "offline-library-test"

        def bind_tools(self, tools, **kwargs):
            assert any(tool["function"]["name"] == "knowledge_library_search" for tool in tools)
            return self

        def _generate(self, messages, **kwargs):
            self._cursor += 1
            if self._cursor == 1:
                response = AIMessage(content="", tool_calls=[{
                    "name": "knowledge_library_search", "id": "library_1",
                    "args": {"query": "Wannier disentanglement", "scope": "local", "limit": 3},
                }])
            else:
                observed.extend(msg.content for msg in messages if isinstance(msg, ToolMessage))
                response = AIMessage(content="Found a local reference; scientific validation remains pending.")
            return ChatResult(generations=[ChatGeneration(message=response)])

    llm = tmp_path / "llm.json"
    llm.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("magnetic_vasp_agent.agent.graph.build_chat_model", lambda _: LibraryModel())
    monkeypatch.setattr("magnetic_vasp_agent.agent.llm_config.load_llm_config", lambda _: LlmConfig(
        model="openai/offline-test", base_url="https://example.invalid/v1",
    ))
    client = TestClient(create_app(root=tmp_path / "web", llm_config=llm), base_url="http://localhost")
    headers = {"X-CSRF-Token": client.get("/api/status").json()["csrf_token"]}
    uploaded = client.post("/api/knowledge/import", headers=headers,
                           data={"scope": "local", "title": "Wannier test reference"},
                           files={"file": ("reference.txt", b"Wannier disentanglement uses a frozen energy window.", "text/plain")})
    assert uploaded.status_code < 300, uploaded.text
    item_id = uploaded.json()["id"]
    confirmed = client.post(f"/api/knowledge/local/{item_id}/index", headers=headers, json={})
    assert confirmed.status_code < 300, confirmed.text
    started = client.post("/api/agent/sessions", headers=headers,
                          json={"goal": "Search the local library for Wannier disentanglement."})
    assert started.status_code == 202, started.text
    detail = client.get(f"/api/agent/sessions/{started.json()['id']}").json()
    assert detail["status"] == "completed", detail
    results = detail["result"]["transcript"]["results"]
    assert results[0]["tool"] == "knowledge.library_search"
    evidence = results[0]["output"]["evidence"]
    assert evidence[0]["item_id"] == item_id
    assert evidence[0]["source_validated"] is False
    assert "frozen energy window" in json.dumps(observed)

    # The same bound callback resolves the new active snapshot, without needing
    # an app restart or handing an arbitrary filesystem path to the model.
    from magnetic_vasp_agent.knowledge.library import KnowledgeLibrary

    registry = ToolRegistry()
    register_library_search(registry, KnowledgeLibrary(tmp_path / "web").search)
    other = client.post("/api/knowledge/import", headers=headers,
                        data={"scope": "local", "title": "Updated local reference"},
                        files={"file": ("updated.txt", b"spinor projection trialorbitals marker2026", "text/plain")}).json()
    client.post(f"/api/knowledge/local/{other['id']}/index", headers=headers, json={}).raise_for_status()
    tool = registry.get("knowledge.library_search")
    response = tool.run(tool.input_model(query="trialorbitals marker2026"))
    assert response["evidence"][0]["item_id"] == other["id"]
