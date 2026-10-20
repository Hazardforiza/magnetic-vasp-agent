from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from magnetic_vasp_agent.agent import tools
from magnetic_vasp_agent.agent.tool_models import RiskTier, ToolSpec
from magnetic_vasp_agent.agent.tools import (
    StructureShowInput,
    ToolRegistry,
    default_registry,
)

EXPECTED_TOOLS = {
    "structure.show": RiskTier.READ_ONLY,
    "incar.show": RiskTier.READ_ONLY,
    "incar.validate": RiskTier.READ_ONLY,
    "kpoints.show": RiskTier.READ_ONLY,
    "kpoints.validate": RiskTier.READ_ONLY,
    "mcif.show": RiskTier.READ_ONLY,
    "potcar.select": RiskTier.READ_ONLY,
    "workflow.validate": RiskTier.READ_ONLY,
    "workflow.inspect": RiskTier.READ_ONLY,
    "workflow.review_card": RiskTier.READ_ONLY,
    "workflow.review_history": RiskTier.READ_ONLY,
    "skill.list": RiskTier.READ_ONLY,
    "skill.show": RiskTier.READ_ONLY,
    "skill.read": RiskTier.READ_ONLY,
    "knowledge.search": RiskTier.READ_ONLY,
    "knowledge.answer": RiskTier.READ_ONLY,
    "execution.probe": RiskTier.READ_ONLY,
    "execution.probe_live": RiskTier.CLUSTER_PLAN,
    "execution.plan": RiskTier.READ_ONLY,
    "execution.inspect_run": RiskTier.READ_ONLY,
    "execution.remote_status": RiskTier.CLUSTER_PLAN,
    "execution.fetch": RiskTier.CLUSTER_PLAN,
    "execution.artifact_release": RiskTier.CLUSTER_EXECUTE,
    "execution.gate_inspect": RiskTier.READ_ONLY,
    "postprocess.parse": RiskTier.READ_ONLY,
    "magnetic.analyze_results": RiskTier.READ_ONLY,
    "postprocess.vaspkit": RiskTier.LOCAL_WRITE,
    "postprocess.vaspkit_catalog": RiskTier.READ_ONLY,
    "postprocess.vaspkit_task": RiskTier.LOCAL_WRITE,
    "postprocess.plot_dos": RiskTier.LOCAL_WRITE,
    "postprocess.plot_band": RiskTier.LOCAL_WRITE,
    "potcar.build": RiskTier.LOCAL_WRITE,
    "incar.build": RiskTier.LOCAL_WRITE,
    "kpoints.build": RiskTier.LOCAL_WRITE,
    "workflow.from_mcif": RiskTier.LOCAL_WRITE,
    "research.from_mcif": RiskTier.LOCAL_WRITE,
    "workflow.build": RiskTier.LOCAL_WRITE,
    "workflow.mae_scan": RiskTier.LOCAL_WRITE,
    "execution.simulate": RiskTier.LOCAL_WRITE,
    "knowledge.evidence": RiskTier.LOCAL_WRITE,
    "knowledge.draft": RiskTier.LOCAL_WRITE,
    "execution.transfer_plan": RiskTier.CLUSTER_PLAN,
    "execution.resource_plan": RiskTier.CLUSTER_PLAN,
    "execution.resource_grant": RiskTier.CLUSTER_PLAN,
    "execution.launch": RiskTier.CLUSTER_EXECUTE,
    "execution.smoke": RiskTier.CLUSTER_EXECUTE,
    "execution.transfer": RiskTier.CLUSTER_EXECUTE,
    "execution.topss_snapshot": RiskTier.CLUSTER_PLAN,
    "execution.request_create": RiskTier.LOCAL_WRITE,
    "execution.transfer_request_create": RiskTier.LOCAL_WRITE,
}


def test_default_registry_has_expected_tools() -> None:
    registry = default_registry()
    assert set(registry.names()) == set(EXPECTED_TOOLS)
    by_name = {spec.name: spec for spec in registry.specs()}
    for name, tier in EXPECTED_TOOLS.items():
        assert by_name[name].risk_tier == tier


def test_only_fixed_cluster_execute_tools_are_registered() -> None:
    registry = default_registry()
    cluster_tools = {
        spec.name
        for spec in registry.specs()
        if spec.risk_tier == RiskTier.CLUSTER_EXECUTE
    }
    assert cluster_tools == {
        "execution.smoke",
        "execution.transfer",
        "execution.launch",
        "execution.artifact_release",
    }


def test_no_shell_like_tool_names_exist() -> None:
    # Guard against tools that would run arbitrary shell or drive the cluster.
    # The read-only "execution.*" preview/plan namespace is explicitly allowed.
    registry = default_registry()
    forbidden = ("shell", "ssh", "mpirun", "cancel", "retry")
    for name in registry.names():
        assert not any(token in name for token in forbidden)
    # Read-only remote operations remain separate from fixed execution tools.
    cluster_tools = {
        spec.name
        for spec in registry.specs()
        if spec.risk_tier == RiskTier.CLUSTER_PLAN
    }
    assert cluster_tools == {
        "execution.transfer_plan",
        "execution.resource_plan",
        "execution.resource_grant",
        "execution.topss_snapshot",
        "execution.probe_live",
        "execution.remote_status",
        "execution.fetch",
    }


def test_registering_cluster_execute_is_rejected() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="cluster_execute"):
        registry.register(
            name="bad.launch",
            description="should never exist",
            risk_tier=RiskTier.CLUSTER_EXECUTE,
            produces_files=False,
            input_model=StructureShowInput,
            run=lambda data: {},
        )


def test_agent_launch_converts_backend_and_forwards_snapshot(monkeypatch, tmp_path) -> None:
    from magnetic_vasp_agent.execution_models import (
        ClusterProfile,
        ExecutionRequest,
        ResourceGrant,
    )
    from magnetic_vasp_agent.execution_runtime_models import LaunchBackend

    captured = {}
    now = datetime.now(UTC)
    profile = ClusterProfile(execution_enabled=True, remote_root="/home/zyy/magvasp/runs")
    request = ExecutionRequest(
        node="cu21", step_id="scf", run_id="run1", mpi_ranks=1, cpu_ids=[0]
    )
    grant = ResourceGrant(
        workflow_id="wf",
        step_id="scf",
        run_id="run1",
        workflow_manifest_sha256="a" * 64,
        step_manifest_sha256="b" * 64,
        run_sha256="c" * 64,
        input_sha256={name: "d" * 64 for name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")},
        node="cu21",
        mpi_ranks=1,
        cpu_ids=[0],
        valid_from=now - timedelta(minutes=1),
        valid_until=now + timedelta(minutes=10),
        approver="zyy",
        approved_at=now,
    )
    profile_path = tmp_path / "profile.json"
    request_path = tmp_path / "request.json"
    grant_path = tmp_path / "grant.json"
    profile_path.write_text(profile.model_dump_json(), encoding="utf-8")
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    grant_path.write_text(grant.model_dump_json(), encoding="utf-8")

    def fake_probe(_profile, _request, _grant, output):
        output.write_text("{}", encoding="utf-8")

    def fake_launch(request):
        captured["request"] = request
        return type(
            "Record",
            (),
            {"model_dump": lambda self, mode="json": {"state": "blocked"}},
        )()

    monkeypatch.setattr(tools, "launch_runtime", fake_launch)
    monkeypatch.setattr(tools, "_collect_bound_action_probe", fake_probe)
    result = tools._run_execution_launch(
        tools.ExecutionLaunchInput(
            backend="remote_mpi",
            confirm="I_CONFIRM_REMOTE_LAUNCH",
            output=str(tmp_path / "run-record"),
            run_id="run1",
            profile=str(profile_path),
            bundle="bundle",
            request=str(request_path),
            transfer_report="transfer.json",
            smoke_report="smoke.json",
            resource_grant=str(grant_path),
            probe_output=str(tmp_path / "probe.json"),
        )
    )
    assert result == {"state": "blocked"}
    assert captured["request"].backend == LaunchBackend.REMOTE_MPI
    assert captured["request"].snapshot == str(tmp_path / "probe.json")


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(
        name="structure.show",
        description="parse a POSCAR",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=StructureShowInput,
        run=lambda data: {},
    )
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(
            name="structure.show",
            description="parse a POSCAR again",
            risk_tier=RiskTier.READ_ONLY,
            produces_files=False,
            input_model=StructureShowInput,
            run=lambda data: {},
        )


def test_get_unknown_tool_raises_keyerror() -> None:
    registry = default_registry()
    with pytest.raises(KeyError, match="unknown tool"):
        registry.get("ssh.run")


def test_gate_inspect_is_local_and_fail_closed(tmp_path) -> None:
    from magnetic_vasp_agent.execution_models import ExecutionState

    state_path = tmp_path / "state.json"
    state_path.write_bytes(ExecutionState(workflow_id="wf", step_id="scf", run_id="r1").model_dump_json().encode())
    tool = default_registry().get("execution.gate_inspect")
    result = tool.run(tool.input_model(state=str(state_path)))
    assert result["next_stage"] == "human_resource_grant"
    assert result["remote_side_effect"] is False
    assert result["cluster_execution"] == "disabled"


def test_specs_are_json_serializable_and_carry_schema() -> None:
    registry = default_registry()
    for spec in registry.specs():
        assert isinstance(spec, ToolSpec)
        payload = spec.model_dump(mode="json")
        assert payload["input_schema"]["additionalProperties"] is False
        assert isinstance(spec.description, str) and spec.description

def test_vaspkit_catalog_reports_versioned_families(tmp_path: Path) -> None:
    from magnetic_vasp_agent.agent.tools import VaspkitCatalogInput
    root = tmp_path / "vaspkit"
    (root / "examples" / "phonon" / "demo").mkdir(parents=True)
    (root / "examples" / "band").mkdir(parents=True)
    result = default_registry().get("postprocess.vaspkit_catalog").run(
        VaspkitCatalogInput(root=str(root))
    )
    assert result["execution_integrated"] is True
    assert {item["name"] for item in result["families"]} == {"band", "phonon"}
    assert result["supported_adapters"]["band_dat"]["task"] == 211


def test_workflow_mae_scan_builds_independent_axis_directories(tmp_path: Path) -> None:
    from magnetic_vasp_agent.agent.tools import WorkflowMaeScanInput
    from magnetic_vasp_agent.models import PotentialPolicy, PotentialRecord, PotentialRegistry
    from magnetic_vasp_agent.workflow import digest

    root = Path(__file__).parents[1]
    poscar = root / "tests" / "fixtures" / "Mn3Al" / "POSCAR"
    library = tmp_path / "potentials"
    records = []
    for element in ["Mn", "Al"]:
        payload = f"SYNTHETIC {element}\n".encode()
        (library / element).mkdir(parents=True)
        (library / element / "POTCAR").write_bytes(payload)
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
    registry = tmp_path / "registry.json"
    registry.write_text(
        PotentialRegistry(root=str(library), family="PBE_54", records=records).model_dump_json(),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(PotentialPolicy(family="PBE_54", preferred={}).model_dump_json(), encoding="utf-8")

    source = json.loads((root / "config" / "workflows" / "Mn3Al.mae.json").read_text(encoding="utf-8"))
    base = source["scans"][0]["base"]
    base["poscar"] = str(poscar)
    request = tmp_path / "base.json"
    request.write_text(json.dumps({"workflow_id": "mn3al", "steps": [base]}), encoding="utf-8")

    output = tmp_path / "mae"
    tool = default_registry().get("workflow.mae_scan")
    result = tool.run(
        WorkflowMaeScanInput(
            request=str(request),
            registry=str(registry),
            policy=str(policy),
            axes={"001": [0, 0, 2], "011": [0, 1, 1]},
            output=str(output),
        )
    )

    assert result["awaiting_human"] is True
    assert result["steps"] == 2
    assert {p.name for p in output.iterdir() if p.is_dir()} == {"mae-001", "mae-011"}
    assert result["axes"]["001"] == [0.0, 0.0, 1.0]
    assert (output / "mae-001" / "INCAR").is_file()
    incar_001 = (output / "mae-001" / "INCAR").read_text(encoding="utf-8")
    incar_011 = (output / "mae-011" / "INCAR").read_text(encoding="utf-8")
    assert "SAXIS = 0.0 0.0 1.0" in incar_001
    assert "SAXIS = 0.0 0.707106781187 0.707106781187" in incar_011
    assert (output / "mae-001" / "POSCAR").read_bytes() == (output / "mae-011" / "POSCAR").read_bytes()
    with pytest.raises(FileExistsError):
        tool.run(
            WorkflowMaeScanInput(
                request=str(request), registry=str(registry), policy=str(policy),
                axes={"001": [0, 0, 1], "011": [0, 1, 1]}, output=str(output),
            )
        )
