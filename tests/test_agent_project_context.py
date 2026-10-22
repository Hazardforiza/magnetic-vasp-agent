from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from magnetic_vasp_agent.agent import tools
from magnetic_vasp_agent.agent.approval import ApprovalPolicy
from magnetic_vasp_agent.agent.project_context import project_registry
from magnetic_vasp_agent.agent.runner import evaluate_call
from magnetic_vasp_agent.agent.tool_models import ApprovalState, ToolCall
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    Cpu,
    ExecutionRequest,
    ProbeSnapshot,
    ProcessObservation,
)


def snapshot(node: str = "node02") -> ProbeSnapshot:
    return ProbeSnapshot(
        source="ssh_probe",
        observed_at=datetime.now(UTC),
        requested_node=node,
        hostname=f"{node}.example.org",
        username="cluster_user",
        cpus=[Cpu(cpu=i, core=i, socket=0, online=True) for i in range(4)],
        allowed_cpu_ids=[0, 1, 2, 3],
        processes=[],
        load1=0,
        mem_available_kib=1024 * 1024,
        module_loaded=True,
        mpi_path="/opt/intel/oneapi/mpi/bin/mpirun",
        executables={"vasp_std": True, "vasp_ncl": True, "vasp_gam": True},
    )


@pytest.fixture
def context(tmp_path: Path) -> SimpleNamespace:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(ClusterProfile().model_dump_json(), encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    work = tmp_path / "planning"
    work.mkdir()
    request_path = work / "execution.request.json"
    request = ExecutionRequest(
        node="node02", step_id="scf", run_id="run1", mpi_ranks=2, cpu_ids=[0, 1]
    )
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    probe_path = work / "node.probe.json"
    registry = project_registry(
        profile_path, bundle, request, request_path=request_path
    )
    return SimpleNamespace(
        profile=profile_path,
        bundle=bundle,
        request=request,
        request_path=request_path,
        work=work,
        probe=probe_path,
        plan=work / "node.resource-plan.json",
        registry=registry,
    )


def evaluate(context: SimpleNamespace, name: str, **arguments):
    return evaluate_call(
        ToolCall(call_id="T1", tool=name, arguments=arguments),
        registry=context.registry,
        policy=ApprovalPolicy(
            approved_cluster_plan_call_ids=["T1"],
            approved_cluster_execute_call_ids=["T1"],
            approver="researcher",
        ),
    )[1]


@pytest.mark.parametrize("name", [
    "execution.request_create",
    "execution.transfer_request_create",
    "execution.resource_grant",
    "execution.smoke",
    "execution.transfer",
    "execution.launch",
    "workflow.build",
    "incar.build",
])
def test_project_planning_cannot_create_requests_or_execute(context, name):
    result = evaluate(context, name)
    assert result.status == "blocked"
    assert result.approval_state == ApprovalState.BLOCKED


def test_project_preview_uses_bound_profile_and_node(context):
    result = evaluate(
        context, "execution.probe", profile=str(context.profile), node="node02"
    )
    assert result.status == "ok"
    assert result.output["mode"] == "readonly_probe_preview"
    assert result.output["executes_mpi"] is False


@pytest.mark.parametrize("node", ["node01", "login01"])
def test_different_node_is_rejected_before_probe(context, monkeypatch, node):
    def unexpected_probe(*args, **kwargs):
        pytest.fail("mismatched node must not reach SSH")

    monkeypatch.setattr(tools, "collect_probe", unexpected_probe)
    result = evaluate(
        context,
        "execution.probe_live",
        profile=str(context.profile),
        node=node,
        output=str(context.probe),
    )
    assert result.status == "validation_error"
    assert "PROJECT_NODE_MISMATCH" in result.message
    assert not context.probe.exists()


@pytest.mark.parametrize("field, value", [
    ("node", "node01"),
    ("mpi_ranks", 1),
    ("cpu_ids", [2, 3]),
    ("step_id", "other-step"),
    ("run_id", "other-run"),
])
def test_saved_request_cannot_change_after_session_binding(context, field, value):
    changed = context.request.model_dump()
    changed[field] = value
    if field == "mpi_ranks":
        changed["cpu_ids"] = None
    context.request_path.write_text(
        ExecutionRequest.model_validate(changed).model_dump_json(), encoding="utf-8"
    )
    result = evaluate(
        context, "execution.probe", profile=str(context.profile), node="node02"
    )
    assert result.status == "validation_error"
    assert "PROJECT_REQUEST_CHANGED" in result.message


@pytest.mark.parametrize("field", ["profile", "bundle", "request"])
def test_execution_plan_cannot_use_other_project_files(context, tmp_path, field):
    arguments = {
        "profile": str(context.profile),
        "bundle": str(context.bundle),
        "request": str(context.request_path),
    }
    arguments[field] = str(tmp_path / "elsewhere")
    result = evaluate(context, "execution.plan", **arguments)
    assert result.status == "validation_error"
    assert f"PROJECT_{field.upper()}_MISMATCH" in result.message


def test_profile_changes_invalidate_bound_session(context):
    context.profile.write_text(
        ClusterProfile(user="someone-else").model_dump_json(), encoding="utf-8"
    )
    result = evaluate(context, "workflow.inspect", bundle=str(context.bundle))
    assert result.status == "validation_error"
    assert "PROJECT_PROFILE_CHANGED" in result.message


def test_project_selection_revision_guard_applies_to_all_tools(context):
    current = {"matches": True}

    def validate_context():
        if not current["matches"]:
            raise ValueError("PROJECT_SELECTION_CHANGED")

    context.registry = project_registry(
        context.profile,
        context.bundle,
        context.request,
        request_path=context.request_path,
        validate_context=validate_context,
    )
    current["matches"] = False
    result = evaluate(context, "skill.list")
    assert result.status == "validation_error"
    assert "PROJECT_SELECTION_CHANGED" in result.message


def test_bound_probe_and_resource_assessment_still_need_human_resource_grant(context, monkeypatch):
    observed = snapshot().model_copy(update={"processes": [ProcessObservation(
        pid=99001, user="other-researcher", last_cpu=2, cpu_percent=0.0,
        command="private-other-calculation",
    )]})
    calls = []

    def collect(profile, node):
        calls.append((profile.user, node))
        return observed

    monkeypatch.setattr(tools, "collect_probe", collect)
    probe = evaluate(
        context,
        "execution.probe_live",
        profile=str(context.profile),
        node="node02",
        output=str(context.probe),
    )
    assert probe.status == "ok"
    assert calls == [("cluster_user", "node02")]
    assert probe.output["mode"] == "project_node_observation"
    assert probe.output["snapshot_path"] == str(context.probe)
    assert probe.output["summary"]["process_count"] == 1
    assert probe.output["summary"]["accessible_physical_cores"] == 4
    public_output = json.dumps(probe.output)
    assert "other-researcher" not in public_output
    assert "private-other-calculation" not in public_output
    assert "99001" not in public_output
    assert "processes" not in probe.output
    persisted = ProbeSnapshot.model_validate_json(context.probe.read_bytes())
    assert persisted == observed
    assert persisted.processes[0].command == "private-other-calculation"
    result = evaluate(
        context,
        "execution.resource_plan",
        profile=str(context.profile),
        request=str(context.request_path),
        snapshot=str(context.probe),
        output=str(context.plan),
    )
    assert result.status == "ok"
    assert result.output["candidate"] == {"node": "node02", "mpi_ranks": 2, "cpu_ids": [0, 1]}
    assert result.output["approved"] is False
    assert result.output["human_review_required"] is True
    assert result.output["assessment"]["ready_to_run"] is False
    assert "ALLOCATION_UNCONFIRMED" in {
        blocker["code"] for blocker in result.output["assessment"]["blockers"]
    }
    assert "private-other-calculation" not in json.dumps(result.output)
    assert "other-researcher" not in json.dumps(result.output)
    assert ExecutionRequest.model_validate_json(context.request_path.read_bytes()) == context.request


def test_cluster_planning_approval_tier_is_retained(context):
    call = ToolCall(
        call_id="T1",
        tool="execution.probe_live",
        arguments={
            "profile": str(context.profile),
            "node": "node02",
            "output": str(context.probe),
        },
    )
    approval, result = evaluate_call(call, registry=context.registry)
    assert approval.state == ApprovalState.AWAITING_APPROVAL
    assert result.status == "blocked"
    assert not context.probe.exists()


def test_other_node_snapshot_cannot_be_used(context):
    context.probe.write_text(snapshot("node01").model_dump_json(), encoding="utf-8")
    result = evaluate(
        context,
        "execution.resource_plan",
        profile=str(context.profile),
        request=str(context.request_path),
        snapshot=str(context.probe),
        output=str(context.plan),
    )
    assert result.status == "validation_error"
    assert "PROJECT_NODE_MISMATCH" in result.message
    assert not context.plan.exists()


@pytest.mark.parametrize("output", ["request", "profile", "bundle", "elsewhere"])
def test_plan_outputs_cannot_replace_project_inputs(context, output):
    context.probe.write_text(snapshot().model_dump_json(), encoding="utf-8")
    paths = {
        "request": context.request_path,
        "profile": context.profile,
        "bundle": context.bundle / "node.resource-plan.json",
        "elsewhere": context.work / "unexpected.json",
    }
    before = context.request_path.read_bytes(), context.profile.read_bytes()
    result = evaluate(
        context,
        "execution.resource_plan",
        profile=str(context.profile),
        request=str(context.request_path),
        snapshot=str(context.probe),
        output=str(paths[output]),
    )
    assert result.status == "validation_error"
    assert "PROJECT_OUTPUT_MISMATCH" in result.message
    assert (context.request_path.read_bytes(), context.profile.read_bytes()) == before
