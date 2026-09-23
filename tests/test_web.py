from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from magnetic_vasp_agent.models import PotentialPolicy, PotentialRecord, PotentialRegistry
from magnetic_vasp_agent import web as web_module
from magnetic_vasp_agent.web import create_app
from magnetic_vasp_agent.workflow import digest

ROOT = Path(__file__).parents[1]
MCIF = ROOT / "tests/fixtures/generated_mcif/candidate_02_FeO_BNS2_7.mcif"
OUTPUTS = ROOT / "tests/fixtures/postprocess/collinear"

TDOS = b"#Energy TDOS-UP TDOS-DOWN\n-1 0.1 -0.2\n0 0.3 -0.4\n1 0.2 -0.1\n"
BAND = b"#K-Path Band-1 Band-2\n0 -2 1\n0.5 -1.5 1.5\n1 -1 2\n"
KLABELS = b"K-Label K-Coordinate\nGAMMA 0\nX 0.5\nM 1\n"


@pytest.fixture
def web_client(tmp_path: Path) -> tuple[TestClient, dict[str, str]]:
    library = tmp_path / "potentials"
    records = []
    for element in ("Fe", "O"):
        source = library / element / "POTCAR"
        source.parent.mkdir(parents=True)
        payload = f"SYNTHETIC TEST POTENTIAL {element}\n".encode()
        source.write_bytes(payload)
        records.append(
            PotentialRecord(
                element=element,
                potential=element,
                family="PBE_54",
                titel=f"Synthetic {element}",
                enmax=400,
                relative_path=f"{element}/POTCAR",
                sha256=digest(payload),
            )
        )
    registry_path = tmp_path / "registry.json"
    policy_path = tmp_path / "policy.json"
    registry_path.write_text(PotentialRegistry(root=str(library), family="PBE_54", records=records).model_dump_json())
    policy_path.write_text(PotentialPolicy(family="PBE_54", preferred={}).model_dump_json())
    app = create_app(root=tmp_path / "web", registry=registry_path, policy=policy_path,
                     llm_config=tmp_path / "llm.json")
    client = TestClient(app, base_url="http://localhost")
    token = client.get("/api/status").json()["csrf_token"]
    return client, {"X-CSRF-Token": token}


def test_local_csrf_and_project_review(web_client: tuple[TestClient, dict[str, str]]) -> None:
    client, headers = web_client
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["project_generation_ready"] is True
    upload = {"file": ("candidate.mcif", MCIF.read_bytes(), "text/plain")}
    assert client.post("/api/projects", files=upload).status_code == 403
    assert client.post("/api/projects", headers={**headers, "Origin": "https://evil.example"}, files=upload).status_code == 403
    assert client.get("/api/status", headers={"Origin": "http://[::1"}).status_code == 403
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 403

    created = client.post("/api/projects", headers=headers, files=upload,
                          data={"task": "static", "include_dos": "true"})
    assert created.status_code == 201, created.text
    detail = created.json()
    assert detail["review_card"]["integrity_valid"] is True
    assert len(detail["steps"]) == 2
    assert all(set(step["inputs"]) == {"POSCAR", "INCAR", "KPOINTS"} for step in detail["steps"])
    assert all(len(step["potcar"]["sha256"]) == 64 for step in detail["steps"])
    assert "SYNTHETIC TEST POTENTIAL" not in str(detail)
    item_id = detail["id"]
    assert client.get(f"/api/projects/{item_id}").status_code == 200
    assert client.get("/api/projects/../../registry.json").status_code != 200
    assert client.get("/api/projects").json()["projects"][0]["id"] == item_id

    reviewed = client.post(f"/api/projects/{item_id}/review", headers=headers,
                           json={"reviewer": "test-reviewer", "decision": "accept"})
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json()["review_card"]["decision"] == "accept"
    assert client.post(f"/api/projects/{item_id}/review", headers=headers,
                       json={"reviewer": "test-reviewer", "decision": "accept"}).status_code == 409


def test_tampered_inputs_cannot_be_accepted(
    web_client: tuple[TestClient, dict[str, str]], tmp_path: Path,
) -> None:
    client, headers = web_client
    created = client.post(
        "/api/projects", headers=headers,
        files={"file": ("candidate.mcif", MCIF.read_bytes(), "text/plain")},
    )
    assert created.status_code == 201, created.text
    item_id = created.json()["id"]
    step = created.json()["steps"][0]["step_id"]
    incar = tmp_path / "web/projects" / item_id / "bundle" / step / "INCAR"
    incar.write_text(incar.read_text(encoding="utf-8") + "\nENCUT = 1\n", encoding="utf-8")

    review = client.post(
        f"/api/projects/{item_id}/review", headers=headers,
        json={"reviewer": "test-reviewer", "decision": "accept"},
    )
    assert review.status_code == 422
    assert "HASH_MISMATCH" in review.json()["detail"]
    card = tmp_path / "web/projects" / item_id / "bundle/input.review.json"
    assert json.loads(card.read_text(encoding="utf-8"))["status"] == "pending"
    assert not (tmp_path / "web/reviews/input-reviews.jsonl").exists()


def test_concurrent_review_records_one_decision(
    web_client: tuple[TestClient, dict[str, str]], tmp_path: Path,
) -> None:
    client, headers = web_client
    created = client.post(
        "/api/projects", headers=headers,
        files={"file": ("candidate.mcif", MCIF.read_bytes(), "text/plain")},
    )
    assert created.status_code == 201, created.text
    item_id = created.json()["id"]

    def decide(decision: str) -> int:
        response = client.post(
            f"/api/projects/{item_id}/review", headers=headers,
            json={"reviewer": "test-reviewer", "decision": decision,
                  "notes": "adjust parameters" if decision == "revise" else None},
        )
        return response.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(decide, ("accept", "revise")))
    assert sorted(statuses) == [200, 409]
    reviews = (tmp_path / "web/reviews/input-reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(reviews) == 1


def test_result_import_parses_and_serves_only_generated_plots(
    web_client: tuple[TestClient, dict[str, str]], tmp_path: Path,
) -> None:
    client, headers = web_client
    files = [
        ("files", ("OSZICAR", (OUTPUTS / "OSZICAR").read_bytes(), "text/plain")),
        ("files", ("OUTCAR", (OUTPUTS / "OUTCAR").read_bytes(), "text/plain")),
        ("files", ("vasprun.xml", (OUTPUTS / "vasprun.xml").read_bytes(), "application/xml")),
        ("files", ("TDOS.dat", TDOS, "text/plain")),
        ("files", ("REFORMATTED_BAND.dat", BAND, "text/plain")),
        ("files", ("KLABELS", KLABELS, "text/plain")),
    ]
    created = client.post("/api/results", headers=headers, files=files)
    assert created.status_code == 201, created.text
    detail = created.json()
    assert detail["summary"]["result_sha256"]
    assert set(detail["plots"]) == {"dos", "band"}
    assert detail["plot_provenance"] == "unverified_external_tables"
    for url in detail["plots"].values():
        image = client.get(url)
        assert image.status_code == 200
        assert image.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert client.get(f"/api/results/{detail['id']}").status_code == 200
    assert client.get("/api/results").json()["results"][0]["id"] == detail["id"]
    metadata = json.loads((tmp_path / "web/results" / detail["id"] / "plots.json").read_text(encoding="utf-8"))
    assert metadata["plot_provenance"] == "unverified_external_tables"
    assert client.get(f"/api/results/{detail['id']}/plots/../../OUTCAR").status_code != 200
    assert client.post("/api/results", headers=headers,
                       files={"files": ("POTCAR", b"secret")}).status_code == 422


def test_failed_uploads_leave_no_managed_entry(
    web_client: tuple[TestClient, dict[str, str]], tmp_path: Path,
) -> None:
    client, headers = web_client
    bad_project = client.post(
        "/api/projects", headers=headers,
        files={"file": ("invalid.mcif", b"not a magnetic CIF", "text/plain")},
    )
    assert bad_project.status_code == 422
    assert client.get("/api/projects").json()["projects"] == []
    assert list((tmp_path / "web/projects").iterdir()) == []

    bad_result = client.post(
        "/api/results", headers=headers,
        files=[
            ("files", ("OSZICAR", (OUTPUTS / "OSZICAR").read_bytes(), "text/plain")),
            ("files", ("TDOS.dat", b"unsupported data", "text/plain")),
        ],
    )
    assert bad_result.status_code == 422
    assert client.get("/api/results").json()["results"] == []
    assert list((tmp_path / "web/results").iterdir()) == []


def test_agent_approval_resume_uses_named_decisions(
    web_client: tuple[TestClient, dict[str, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, headers = web_client
    llm = tmp_path / "llm.json"
    llm.write_text("{}")
    calls: list[dict] = []

    class FakeResult:
        def __init__(self, pending: bool) -> None:
            self.pending_approvals = [
                SimpleNamespace(call_id="T1", model_dump=lambda **_: {
                    "call_id": "T1", "tool": "execution.resource_plan",
                    "risk_tier": "cluster_plan", "arguments": {}, "reason": "review",
                })
            ] if pending else []

        def model_dump(self, **_: object) -> dict:
            return {"pending_approvals": [item.model_dump() for item in self.pending_approvals]}

    class FakeGraph:
        def __init__(self, config: object) -> None:
            pass

        def run(self, goal: str) -> FakeResult:
            assert goal == "plan a calculation"
            return FakeResult(True)

        def resume(self, previous: FakeResult, decisions: dict) -> FakeResult:
            calls.append(decisions)
            return FakeResult(False)

    monkeypatch.setattr("magnetic_vasp_agent.agent.graph.VaspAgentGraph", FakeGraph)
    monkeypatch.setattr("magnetic_vasp_agent.agent.llm_config.load_llm_config", lambda _: object())
    created = client.post("/api/agent/sessions", headers=headers, json={"goal": "plan a calculation"})
    assert created.status_code == 202, created.text
    item_id = created.json()["id"]
    assert client.get(f"/api/agent/sessions/{item_id}").json()["status"] == "pending_approval"
    assert client.post(f"/api/agent/sessions/{item_id}/decisions", headers=headers,
                       json={"reviewer": "agent", "decisions": {"T1": True}}).status_code == 422
    assert client.post(f"/api/agent/sessions/{item_id}/decisions", headers=headers,
                       json={"reviewer": "test-reviewer", "decisions": {"T2": True}}).status_code == 422
    approved = client.post(f"/api/agent/sessions/{item_id}/decisions", headers=headers,
                           json={"reviewer": "test-reviewer", "decisions": {"T1": True}})
    assert approved.status_code == 202, approved.text
    assert client.get(f"/api/agent/sessions/{item_id}").json()["status"] == "completed"
    assert calls == [{"T1": {"approve": True, "approver": "test-reviewer"}}]


def test_completed_agent_sessions_are_bounded(
    web_client: tuple[TestClient, dict[str, str]], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, headers = web_client
    (tmp_path / "llm.json").write_text("{}")
    monkeypatch.setattr(web_module, "_MAX_SESSIONS", 2)

    class FakeResult:
        pending_approvals: list = []

        def model_dump(self, **_: object) -> dict:
            return {"pending_approvals": []}

    class FakeGraph:
        def __init__(self, config: object) -> None:
            pass

        def run(self, goal: str) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr("magnetic_vasp_agent.agent.graph.VaspAgentGraph", FakeGraph)
    monkeypatch.setattr("magnetic_vasp_agent.agent.llm_config.load_llm_config", lambda _: object())
    ids = []
    for number in range(3):
        response = client.post("/api/agent/sessions", headers=headers,
                               json={"goal": f"run {number}"})
        assert response.status_code == 202
        ids.append(response.json()["id"])
    assert client.get(f"/api/agent/sessions/{ids[0]}").status_code == 404
    assert all(client.get(f"/api/agent/sessions/{item_id}").status_code == 200
               for item_id in ids[1:])
