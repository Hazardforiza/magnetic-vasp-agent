"""Offline regressions for the resource-tool integration; no SSH is invoked."""

from types import SimpleNamespace

from magnetic_vasp_agent.agent import tools
from magnetic_vasp_agent.execution_models import ClusterProfile, ExecutionRequest
from magnetic_vasp_agent import execution_runtime as runtime
from magnetic_vasp_agent.execution_runtime_models import LaunchRequest, LaunchBackend, REMOTE_CONFIRM


def test_agent_execution_plan_returns_prepared_report(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    request = tmp_path / "request.json"
    profile.write_text(ClusterProfile().model_dump_json())
    request.write_text(ExecutionRequest(node="node04", step_id="scf", run_id="r1").model_dump_json())
    report = {"mode": "execution_preview", "ready_to_run": False}

    def prepare(bundle, settings, intent, snapshot):
        assert intent.node == "node04"
        assert snapshot is None
        return report

    monkeypatch.setattr(tools, "prepare_execution", prepare)
    result = tools._run_execution_plan(tools.ExecutionPlanInput(
        bundle=str(tmp_path), profile=str(profile), request=str(request),
    ))
    assert result == report


def test_launch_checks_bundle_before_grant_workflow_match(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    request = tmp_path / "request.json"
    smoke = tmp_path / "smoke.json"
    transfer = tmp_path / "transfer.json"
    grant_file = tmp_path / "grant.json"
    profile.write_text(ClusterProfile(execution_enabled=True, remote_root="/home/cluster_user/magvasp/runs").model_dump_json())
    request.write_text(ExecutionRequest(node="node04", step_id="scf", run_id="r1").model_dump_json())
    smoke.write_text('{"passed":true,"mode":"mpi_smoke","node":"node04"}')
    transfer.write_text('{}')
    grant_file.write_text('{}')
    monkeypatch.setattr(runtime, "transfer_report_model", lambda _: SimpleNamespace(node="node04"))
    monkeypatch.setattr(runtime, "ResourceGrant", SimpleNamespace(
        model_validate_json=lambda _: SimpleNamespace(matches_request=lambda _: True, workflow_id="other"),
    ))
    monkeypatch.setattr(runtime, "_read_step", lambda *_: ({"workflow_id": "wf"}, {}, None))

    def must_not_launch(*args, **kwargs):
        raise AssertionError("mismatched grant must never reach remote execution")

    monkeypatch.setattr(runtime, "start_remote_vasp", must_not_launch)
    result = runtime.launch(LaunchRequest(
        backend=LaunchBackend.REMOTE_MPI, confirm=REMOTE_CONFIRM,
        output=str(tmp_path / "record"), run_id="r1", bundle=str(tmp_path),
        profile=str(profile), request=str(request), smoke_report=str(smoke),
        transfer_report=str(transfer), resource_grant=str(grant_file),
    ))
    assert result.blockers == ["RESOURCE_GRANT_WORKFLOW_MISMATCH"]
