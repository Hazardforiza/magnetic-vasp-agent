from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from magnetic_vasp_agent.models import PotentialPolicy, PotentialRecord, PotentialRegistry
from magnetic_vasp_agent.execution_models import Cpu, ClusterProfile, ExecutionRequest, ProcessObservation, ProbeSnapshot, WannierProbeSnapshot
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
    execution_profile_path = tmp_path / "cluster.json"
    registry_path.write_text(PotentialRegistry(root=str(library), family="PBE_54", records=records).model_dump_json())
    policy_path.write_text(PotentialPolicy(family="PBE_54", preferred={}).model_dump_json())
    execution_profile_path.write_text(ClusterProfile(execution_enabled=False).model_dump_json())
    app = create_app(root=tmp_path / "web", registry=registry_path, policy=policy_path,
                     llm_config=tmp_path / "llm.json", execution_profile=execution_profile_path)
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
                           json={"reviewer": "cluster_user", "decision": "accept"})
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json()["review_card"]["decision"] == "accept"
    assert client.post(f"/api/projects/{item_id}/review", headers=headers,
                       json={"reviewer": "cluster_user", "decision": "accept"}).status_code == 409


@pytest.mark.parametrize(
    ("task", "expected_prefix", "minimum_steps"),
    [
        ("magnetic_order", "static-", 2),
        ("convergence", "conv-", 2),
        ("mae", "mae-", 3),
    ],
)
def test_research_tasks_generate_compilable_project_steps(
    web_client: tuple[TestClient, dict[str, str]], task: str, expected_prefix: str,
    minimum_steps: int,
) -> None:
    client, headers = web_client
    response = client.post(
        "/api/projects",
        headers=headers,
        files={"file": ("candidate.mcif", MCIF.read_bytes(), "text/plain")},
        data={"task": task},
    )
    assert response.status_code == 201, response.text
    detail = response.json()
    step_ids = [step["step_id"] for step in detail["steps"]]
    assert len(step_ids) >= minimum_steps
    assert all(step_id.startswith(expected_prefix) for step_id in step_ids)
    assert detail["review_card"]["integrity_valid"] is True


def test_project_dos_is_limited_to_static(web_client: tuple[TestClient, dict[str, str]]) -> None:
    client, headers = web_client
    response = client.post(
        "/api/projects",
        headers=headers,
        files={"file": ("candidate.mcif", MCIF.read_bytes(), "text/plain")},
        data={"task": "convergence", "include_dos": "true"},
    )
    assert response.status_code == 422
    assert "DOS follow-up" in response.json()["detail"]


def test_node_inventory_is_read_only_and_excludes_gateway(
    web_client: tuple[TestClient, dict[str, str]],
) -> None:
    client, _ = web_client
    inventory = client.get("/api/nodes")
    assert inventory.status_code == 200
    payload = inventory.json()
    assert payload["configured"] is True
    assert "gateway_hostname" not in payload
    assert "execution_enabled" not in payload
    names = [item["name"] for item in payload["nodes"]]
    assert "login01" not in names
    assert names == list(ClusterProfile().allowed_nodes)
    assert all(item["status"] == "unknown" for item in payload["nodes"])
    assert payload["observation_note"]
    assert client.get("/api/nodes/login01/probe").status_code == 404


def test_node_inventory_reports_missing_profile(tmp_path: Path) -> None:
    app = create_app(
        root=tmp_path / "web",
        execution_profile=tmp_path / "missing-cluster.json",
    )
    client = TestClient(app, base_url="http://localhost")
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["execution_profile_configured"] is False
    inventory = client.get("/api/nodes")
    assert inventory.status_code == 200
    assert inventory.json()["configured"] is False
    assert inventory.json()["nodes"] == []
    assert client.get("/api/nodes/node01/probe").status_code == 503


def test_node_probe_redacts_processes_and_reports_wannier_capability(
    web_client: tuple[TestClient, dict[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = web_client
    snapshot = ProbeSnapshot(
        source="offline_fixture", observed_at=datetime.now(UTC),
        requested_node="node01", hostname="node01", username="cluster_user",
        cpus=[Cpu(cpu=0, core=0, socket=0, online=True)], allowed_cpu_ids=[0],
        processes=[ProcessObservation(pid=10, user="other", last_cpu=0, cpu_percent=0.1, command="secret-command")],
        load1=0.0, mem_available_kib=2 * 1024 * 1024, module_loaded=True,
        mpi_path=ClusterProfile().mpi_path,
        executables={"vasp_std": True, "vasp_ncl": True, "vasp_gam": True},
    )
    wannier = WannierProbeSnapshot(
        source="offline_fixture", observed_at=snapshot.observed_at, requested_node="node01",
        hostname="node01", username="cluster_user", module_name=ClusterProfile().wannier_module,
        module_loaded=True, wannier90_path="/opt/wannier90/3.1.0/bin/wannier90.x",
        wannier90_version="3.1.0", vasp_ncl_path="/opt/vasp/bin/vasp_ncl",
    )
    monkeypatch.setattr(web_module, "collect_probe", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(web_module, "collect_wannier_probe", lambda *args, **kwargs: wannier)
    response = client.get("/api/nodes/node01/probe")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "observable"
    assert payload["snapshot"]["cpu_count"] == 1
    assert payload["wannier"]["status"] == "verified"
    assert payload["wannier"]["version"] == "3.1.0"
    assert "processes" not in payload
    assert "secret-command" not in response.text


def test_node_probe_failure_is_a_safe_status(
    web_client: tuple[TestClient, dict[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = web_client
    monkeypatch.setattr(
        web_module, "collect_probe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("SSH_PROBE_FAILED: remote-secret-command")
        ),
    )
    response = client.get("/api/nodes/node01/probe")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "connection_failed"
    assert payload["error"]["code"] == "SSH_PROBE_FAILED"
    assert "remote-secret-command" not in response.text


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
        json={"reviewer": "cluster_user", "decision": "accept"},
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
            json={"reviewer": "cluster_user", "decision": decision,
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
        def __init__(self, config: object, *, registry) -> None:
            assert "knowledge.library_search" in registry

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
                       json={"reviewer": "cluster_user", "decisions": {"T2": True}}).status_code == 422
    approved = client.post(f"/api/agent/sessions/{item_id}/decisions", headers=headers,
                           json={"reviewer": "cluster_user", "decisions": {"T1": True}})
    assert approved.status_code == 202, approved.text
    assert client.get(f"/api/agent/sessions/{item_id}").json()["status"] == "completed"
    assert calls == [{"T1": {"approve": True, "approver": "cluster_user"}}]


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
        def __init__(self, config: object, *, registry) -> None:
            assert "knowledge.library_search" in registry

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


def _create_resource_project(client, headers, **data):
    return client.post("/api/projects", headers=headers,
                       files={"file": ("candidate.mcif", MCIF.read_bytes(), "text/plain")},
                       data=data)


def test_project_resources_persist_without_changing_input_review(web_client, tmp_path):
    client, headers = web_client
    response = _create_resource_project(client, headers, resource_mode="manual",
                                        resource_node="node01", resource_mpi_ranks="8")
    assert response.status_code == 201, response.text
    detail = response.json()
    item_id = detail["id"]
    resources = detail["resources"]
    assert (resources["mode"], resources["node"], resources["mpi_ranks"]) == ("manual", "node01", 8)
    assert resources["selection"] is None
    accepted = client.post(f"/api/projects/{item_id}/review", headers=headers,
                           json={"reviewer": "cluster_user", "decision": "accept"}).json()
    changed = client.put(f"/api/projects/{item_id}/resources", headers=headers,
                         json={"mode": "manual", "node": "node02", "mpi_ranks": 16})
    assert changed.status_code == 200, changed.text
    assert changed.json()["resources"]["revision"] > resources["revision"]
    assert changed.json()["review_card"] == accepted["review_card"]
    restarted = TestClient(create_app(root=tmp_path / "web"), base_url="http://localhost")
    assert restarted.get(f"/api/projects/{item_id}").json()["resources"] == changed.json()["resources"]
    (tmp_path / "web/projects" / item_id / "resources.json").unlink()
    assert restarted.get(f"/api/projects/{item_id}").json()["resources"]["mode"] == "llm"


@pytest.mark.parametrize("preferences", [
    {"mode": "manual", "node": "login01", "mpi_ranks": 8},
    {"mode": "manual", "node": "cu99", "mpi_ranks": 8},
    {"mode": "manual", "node": "node01", "mpi_ranks": 0},
    {"mode": "manual", "node": "node01", "mpi_ranks": 1025},
    {"mode": "manual", "node": "node01", "mpi_ranks": 1.5},
    {"mode": "manual", "node": "node01", "mpi_ranks": True},
    {"mode": "manual", "node": "node01"},
    {"mode": "llm", "node": "node01", "mpi_ranks": 8},
])
def test_invalid_project_resource_preferences_are_rejected(web_client, preferences):
    client, headers = web_client
    detail = _create_resource_project(client, headers).json()
    item_id = detail["id"]
    result = client.put(f"/api/projects/{item_id}/resources", headers=headers, json=preferences)
    assert result.status_code == 422, result.text
    assert client.get(f"/api/projects/{item_id}").json()["resources"] == detail["resources"]


def test_invalid_resource_form_does_not_allocate_project(web_client, tmp_path):
    client, headers = web_client
    for data in ({"resource_mode": "manual", "resource_node": "login01", "resource_mpi_ranks": "4"},
                 {"resource_mode": "manual"},
                 {"resource_mode": "manual", "resource_node": "node01", "resource_mpi_ranks": "1.5"}):
        result = _create_resource_project(client, headers, **data)
        assert result.status_code == 422, result.text
    assert list((tmp_path / "web/projects").iterdir()) == []


def test_resource_selection_persists_and_project_agent_is_bound(web_client, tmp_path, monkeypatch):
    client, headers = web_client
    detail = _create_resource_project(client, headers, resource_mode="manual",
                                     resource_node="node01", resource_mpi_ranks="2").json()
    item_id = detail["id"]
    (tmp_path / "llm.json").write_text("{}")
    assert client.post("/api/agent/sessions", headers=headers,
                       json={"goal": "plan", "project_id": item_id}).status_code == 409
    selected_requests = []

    def choose(profile, preferences, context, llm_path):
        assert preferences.node == "node01" and preferences.mpi_ranks == 2
        assert context["executable"] == detail["review_card"]["steps"][0]["executable"]
        assert context["atom_count"] > 0
        request = ExecutionRequest(node=preferences.node, mpi_ranks=preferences.mpi_ranks,
                                   cpu_ids=[0, 1], step_id=context["step_id"], run_id=context["run_id"])
        selected_requests.append(request)
        return {"source": "manual", "node": request.node, "mpi_ranks": request.mpi_ranks,
                "execution_request": request.model_dump(mode="json")}

    monkeypatch.setattr(web_module, "choose_project_resources", choose)
    selection = client.post(f"/api/projects/{item_id}/resources/select", headers=headers)
    assert selection.status_code == 200, selection.text
    selected = selection.json()["resources"]
    assert client.get(f"/api/projects/{item_id}").json()["resources"] == selected
    assert selected["selection"]["node"] == "node01"
    bound_registries = []

    class FakeGraph:
        def __init__(self, config, *, registry):
            bound_registries.append(registry)

        def run(self, goal):
            assert "node=node01; mpi_ranks=2" in goal
            return SimpleNamespace(pending_approvals=[], model_dump=lambda **_: {})

    monkeypatch.setattr("magnetic_vasp_agent.agent.graph.VaspAgentGraph", FakeGraph)
    monkeypatch.setattr("magnetic_vasp_agent.agent.llm_config.load_llm_config", lambda _: object())
    session = client.post("/api/agent/sessions", headers=headers,
                          json={"goal": "plan", "project_id": item_id})
    assert session.status_code == 202, session.text
    assert client.get(f"/api/agent/sessions/{session.json()['id']}").json()["status"] == "completed"
    registry = bound_registries[0]
    assert "execution.resource_plan" in registry.names()
    assert not {"execution.launch", "execution.smoke", "execution.transfer", "execution.resource_grant"}.intersection(registry.names())
    profile_file = tmp_path / "cluster.json"
    original_profile = profile_file.read_bytes()
    modified_profile = json.loads(original_profile)
    modified_profile["user"] = "another-user"
    profile_file.write_text(json.dumps(modified_profile))
    assert client.post("/api/agent/sessions", headers=headers,
                       json={"goal": "plan", "project_id": item_id}).status_code == 409
    preview = registry.get("execution.probe")
    with pytest.raises(ValueError, match="PROJECT_PROFILE_CHANGED"):
        preview.run(preview.input_model(profile=str(profile_file), node="node01"))
    profile_file.write_bytes(original_profile)
    changed = client.put(f"/api/projects/{item_id}/resources", headers=headers,
                         json={"mode": "manual", "node": "node02", "mpi_ranks": 4})
    assert changed.status_code == 200
    assert changed.json()["resources"]["selection"] is None
    with pytest.raises(ValueError, match="PROJECT_RESOURCES_CHANGED"):
        preview.run(preview.input_model(profile=str(tmp_path / "cluster.json"), node="node01"))


def test_resource_selection_failure_is_explicit_and_does_not_save_choice(web_client, monkeypatch):
    from magnetic_vasp_agent.project_resources import ResourceSelectionError

    client, headers = web_client
    detail = _create_resource_project(client, headers).json()
    item_id = detail["id"]

    def fail(*_):
        raise ResourceSelectionError("NO_ELIGIBLE_NODES", "无法取得有效节点观察。")

    monkeypatch.setattr(web_module, "choose_project_resources", fail)
    response = client.post(f"/api/projects/{item_id}/resources/select", headers=headers)
    assert response.status_code == 422
    assert "NO_ELIGIBLE_NODES" in response.json()["detail"]
    assert client.get(f"/api/projects/{item_id}").json()["resources"]["selection"] is None


def test_profile_change_during_selection_discards_result_and_blocks_overlapping_updates(web_client, tmp_path, monkeypatch):
    client, headers = web_client
    detail = _create_resource_project(client, headers).json()
    item_id = detail["id"]
    resource_url = f"/api/projects/{item_id}/resources"

    def choose(profile, preferences, context, llm_path):
        assert client.put(resource_url, headers=headers, json={"mode": "llm"}).status_code == 409
        assert client.post(resource_url + "/select", headers=headers).status_code == 409
        (tmp_path / "cluster.json").write_text(profile.model_copy(update={"user": "changed"}).model_dump_json())
        return {"node": "node01", "mpi_ranks": 2}

    monkeypatch.setattr(web_module, "choose_project_resources", choose)
    response = client.post(resource_url + "/select", headers=headers)
    assert response.status_code == 409
    assert "配置已更改" in response.json()["detail"]
    assert client.get(f"/api/projects/{item_id}").json()["resources"]["selection"] is None
    # The in-progress marker must be released even when the observation is discarded.
    assert client.put(resource_url, headers=headers, json={"mode": "llm"}).status_code == 200
