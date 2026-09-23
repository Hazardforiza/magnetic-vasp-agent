from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from magnetic_vasp_agent.agent import tools
from magnetic_vasp_agent.agent.approval import ApprovalPolicy
from magnetic_vasp_agent.agent.runner import evaluate_call
from magnetic_vasp_agent.agent.tool_models import ApprovalState, ToolCall
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    Cpu,
    ExecutionRequest,
    ProbeSnapshot,
    ResourceGrant,
)
from magnetic_vasp_agent.transfer_models import TransferRequest


def _write(path: Path, model: object) -> None:
    path.write_text(model.model_dump_json(), encoding="utf-8")  # type: ignore[attr-defined]


def _snapshot(observed_at: datetime, *, node: str = "cu21") -> ProbeSnapshot:
    return ProbeSnapshot(
        source="ssh_probe",
        observed_at=observed_at,
        requested_node=node,
        hostname=f"{node}.cmphc",
        username="cluster_user",
        cpus=[Cpu(cpu=0, core=0, socket=0, online=True)],
        allowed_cpu_ids=[0],
        processes=[],
        load1=0,
        mem_available_kib=1024,
        module_loaded=True,
        mpi_path="/public/intel/oneapi/mpi/2021.1.1/bin/mpirun",
        executables={"vasp_std": True, "vasp_ncl": True, "vasp_gam": True},
    )


def _grant(now: datetime) -> ResourceGrant:
    return ResourceGrant(
        workflow_id="wf",
        step_id="scf",
        run_id="run1",
        workflow_manifest_sha256="a" * 64,
        step_manifest_sha256="b" * 64,
        run_sha256="c" * 64,
        input_sha256={
            name: "d" * 64 for name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
        },
        node="cu21",
        mpi_ranks=1,
        cpu_ids=[0],
        valid_from=now - timedelta(seconds=10),
        valid_until=now + timedelta(minutes=30),
        approver="test-reviewer",
        approved_at=now - timedelta(seconds=10),
    )


def _control_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    now = datetime.now(UTC)
    profile_path = tmp_path / "profile.json"
    request_path = tmp_path / "request.json"
    grant_path = tmp_path / "grant.json"
    _write(
        profile_path,
        ClusterProfile(execution_enabled=True, remote_root="/home/cluster_user/magvasp/runs"),
    )
    _write(
        request_path,
        ExecutionRequest(
            node="cu21", step_id="scf", run_id="run1", mpi_ranks=1, cpu_ids=[0]
        ),
    )
    _write(grant_path, _grant(now))
    return profile_path, request_path, grant_path


def test_agent_creates_typed_execution_and_transfer_requests(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    _write(profile, ClusterProfile())

    execution_output = tmp_path / "execution-request.json"
    result = tools._run_execution_request_create(
        tools.ExecutionRequestCreateInput(
            profile=str(profile),
            node="cu21",
            step_id="scf",
            run_id="run1",
            mpi_ranks=2,
            cpu_ids=[0, 1],
            output=str(execution_output),
        )
    )
    request = ExecutionRequest.model_validate_json(execution_output.read_bytes())
    assert request.cpu_ids == [0, 1]
    assert result["sha256"]

    transfer_output = tmp_path / "transfer-request.json"
    tools._run_execution_transfer_request_create(
        tools.ExecutionTransferRequestCreateInput(
            profile=str(profile),
            execution_request=str(execution_output),
            output=str(transfer_output),
        )
    )
    transfer = TransferRequest.model_validate_json(transfer_output.read_bytes())
    assert transfer.node == request.node
    assert transfer.run_id == request.run_id
    assert transfer.overwrite is False


def test_request_creation_rejects_nonallowlisted_node(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    _write(profile, ClusterProfile())
    with pytest.raises(ValueError, match="NODE_FORBIDDEN"):
        tools._run_execution_request_create(
            tools.ExecutionRequestCreateInput(
                profile=str(profile),
                node="cu42",
                step_id="scf",
                run_id="run1",
                output=str(tmp_path / "request.json"),
            )
        )


def test_resource_grant_requires_cluster_plan_and_same_named_human() -> None:
    arguments = {
        "bundle": "bundle",
        "profile": "profile.json",
        "request": "request.json",
        "approver": "test-reviewer",
        "confirm": "I_APPROVE_RESOURCE_GRANT",
        "probe_output": "probe.json",
        "output": "grant.json",
    }
    call = ToolCall(call_id="T1", tool="execution.resource_grant", arguments=arguments)

    approval, result = evaluate_call(call)
    assert approval.state == ApprovalState.AWAITING_APPROVAL
    assert result.status == "blocked"

    approval, result = evaluate_call(
        call,
        policy=ApprovalPolicy(
            approved_cluster_plan_call_ids=["T1"], approver="someone-else"
        ),
    )
    assert approval.state == ApprovalState.BLOCKED
    assert result.status == "blocked"
    assert "must match" in (result.message or "")


def test_resource_grant_adapter_probes_after_exact_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path, request_path, _ = _control_files(tmp_path)
    now = datetime.now(UTC)
    snapshot = _snapshot(now)
    grant = _grant(now)
    events: list[str] = []

    def fake_probe(profile, node):
        del profile
        assert node == "cu21"
        events.append("probe")
        return snapshot

    def fake_issue(bundle, profile, request, observed, output, **kwargs):
        del bundle, profile, request
        assert observed is snapshot
        assert kwargs["approver"] == "test-reviewer"
        assert kwargs["confirm"] == "I_APPROVE_RESOURCE_GRANT"
        events.append("grant")
        output.write_text(grant.model_dump_json(), encoding="utf-8")
        return grant

    monkeypatch.setattr(tools, "collect_probe", fake_probe)
    monkeypatch.setattr(tools, "issue_resource_grant", fake_issue)
    probe_output = tmp_path / "grant-probe.json"
    grant_output = tmp_path / "resource-grant.json"
    result = tools._run_execution_resource_grant(
        tools.ExecutionResourceGrantInput(
            bundle=str(tmp_path / "bundle"),
            profile=str(profile_path),
            request=str(request_path),
            approver="test-reviewer",
            confirm="I_APPROVE_RESOURCE_GRANT",
            probe_output=str(probe_output),
            output=str(grant_output),
        )
    )

    assert events == ["probe", "grant"]
    assert ProbeSnapshot.model_validate_json(probe_output.read_bytes()) == snapshot
    assert result["probe_snapshot_sha256"]


def test_bad_grant_phrase_fails_before_any_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tools,
        "collect_probe",
        lambda *_args, **_kwargs: pytest.fail("probe must not run"),
    )
    with pytest.raises(ValueError, match="I_APPROVE_RESOURCE_GRANT"):
        tools._run_execution_resource_grant(
            tools.ExecutionResourceGrantInput(
                bundle="bundle",
                profile="profile.json",
                request="request.json",
                approver="test-reviewer",
                confirm="yes",
                probe_output="probe.json",
                output="grant.json",
            )
        )


def test_smoke_collects_and_validates_action_local_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, request, grant = _control_files(tmp_path)
    events: list[str] = []

    def fake_probe(_profile, _node):
        events.append("probe")
        return _snapshot(datetime.now(UTC))

    def fake_smoke(*_args, **_kwargs):
        events.append("smoke")
        return {"mode": "mpi_smoke", "passed": True}

    monkeypatch.setattr(tools, "collect_probe", fake_probe)
    monkeypatch.setattr(tools, "run_mpi_smoke", fake_smoke)
    result = tools._run_execution_smoke(
        tools.ExecutionSmokeInput(
            profile=str(profile),
            request=str(request),
            resource_grant=str(grant),
            confirm="I_CONFIRM_MPI_SMOKE",
            probe_output=str(tmp_path / "smoke-probe.json"),
            output=str(tmp_path / "smoke.json"),
        )
    )

    assert result["passed"] is True
    assert events == ["probe", "smoke"]


def test_stale_action_local_probe_blocks_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, request, grant = _control_files(tmp_path)
    monkeypatch.setattr(
        tools,
        "collect_probe",
        lambda *_args: _snapshot(datetime.now(UTC) - timedelta(seconds=61)),
    )
    monkeypatch.setattr(
        tools,
        "run_mpi_smoke",
        lambda *_args, **_kwargs: pytest.fail("smoke must stay blocked"),
    )
    with pytest.raises(ValueError, match="SNAPSHOT_STALE"):
        tools._run_execution_smoke(
            tools.ExecutionSmokeInput(
                profile=str(profile),
                request=str(request),
                resource_grant=str(grant),
                confirm="I_CONFIRM_MPI_SMOKE",
                probe_output=str(tmp_path / "stale-probe.json"),
                output=str(tmp_path / "smoke.json"),
            )
        )


def test_fetch_refuses_existing_report_before_remote_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fetch-report.json"
    output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        tools,
        "fetch_results",
        lambda *_args, **_kwargs: pytest.fail("remote fetch must not run"),
    )

    with pytest.raises(FileExistsError, match="output exists"):
        tools._run_execution_fetch(
            tools.ExecutionFetchInput(
                profile="missing-profile.json",
                run="missing-run",
                destination=str(tmp_path / "results"),
                files=["OUTCAR"],
                output=str(output),
            )
        )
