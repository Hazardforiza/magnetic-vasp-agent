from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from magnetic_vasp_agent import execution_runtime as runtime
from magnetic_vasp_agent.execution_approval import (
    RESOURCE_GRANT_CONFIRM,
    build_resource_grant,
)
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    Cpu,
    ExecutionRequest,
    ProbeSnapshot,
    ResourceGrant,
    SmokeTestEvidence,
)
from magnetic_vasp_agent.execution_runtime_models import (
    REMOTE_CONFIRM,
    JobState,
    LaunchBackend,
    LaunchRequest,
)
from magnetic_vasp_agent.input_review import build_review_card, record_input_review
from magnetic_vasp_agent.input_review_models import InputReviewCard
from magnetic_vasp_agent.mcif_workflow import plan_workflow_from_mcif
from magnetic_vasp_agent.models import PotentialPolicy, PotentialRecord, PotentialRegistry
from magnetic_vasp_agent.transfer import build_transfer_plan, transfer_report_model
from magnetic_vasp_agent.transfer_models import TransferFile, TransferReport, TransferRequest
from magnetic_vasp_agent.workflow import json_bytes, materialize_workflow, prepare_workflow
from magnetic_vasp_agent.workflow_models import Workflow

ROOT = Path(__file__).parents[1]
ZERO_HASH = "0" * 64


@dataclass
class EvidenceContext:
    bundle: Path
    profile: ClusterProfile
    execution: ExecutionRequest
    snapshot: ProbeSnapshot
    grant: ResourceGrant
    smoke: SmokeTestEvidence
    transfer: TransferReport
    review: InputReviewCard
    launch_request: LaunchRequest
    paths: dict[str, Path]


def _write_model(path: Path, model: object) -> None:
    payload = model.model_dump(mode="json")  # type: ignore[attr-defined]
    path.write_bytes(json_bytes(payload))


def _snapshot(observed_at: datetime, *, node: str = "node02") -> ProbeSnapshot:
    cpus = [
        Cpu(cpu=index, core=index % 20, socket=(index % 20) // 10, online=True)
        for index in range(40)
    ]
    return ProbeSnapshot(
        source="ssh_probe",
        observed_at=observed_at,
        requested_node=node,
        hostname=f"{node}.example.org",
        username="cluster_user",
        cpus=cpus,
        allowed_cpu_ids=list(range(40)),
        processes=[],
        load1=0.0,
        mem_available_kib=8 * 1024 * 1024,
        module_loaded=True,
        mpi_path="/opt/intel/oneapi/mpi/bin/mpirun",
        executables={"vasp_std": True, "vasp_ncl": True, "vasp_gam": True},
    )


@pytest.fixture
def evidence_context(tmp_path: Path) -> EvidenceContext:
    potential_records = []
    for element in ("Fe", "O"):
        payload = f"Synthetic test potential: {element}\n".encode()
        path = tmp_path / "potentials" / element / "POTCAR"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        potential_records.append(
            PotentialRecord(
                element=element,
                potential=element,
                family="PBE_54",
                titel=f"Synthetic {element}",
                relative_path=f"{element}/POTCAR",
                sha256=hashlib.sha256(payload).hexdigest(),
                enmax=400,
            )
        )

    source = ROOT / "tests/fixtures/generated_mcif/candidate_01_FeO_BNS2_7.mcif"
    workflow_request = tmp_path / "workflow.json"
    plan_workflow_from_mcif(
        [source], workflow_request, workflow_id="runtime-evidence"
    )
    prepared = prepare_workflow(
        Workflow.model_validate_json(workflow_request.read_bytes()),
        PotentialRegistry(
            root=str(tmp_path / "potentials"),
            family="PBE_54",
            records=potential_records,
        ),
        PotentialPolicy(family="PBE_54", preferred={}),
        tmp_path,
    )
    bundle = tmp_path / "bundle"
    materialize_workflow(prepared, bundle)
    build_review_card(bundle, log_path=tmp_path / "reviews.jsonl")
    review = record_input_review(
        bundle,
        reviewer="cluster_user",
        decision="accept",
        log_path=tmp_path / "reviews.jsonl",
    )

    manifest = json.loads((bundle / "workflow.manifest.json").read_bytes())
    step_id = manifest["order"][0]
    profile = ClusterProfile(
        execution_enabled=True, remote_root="/home/cluster_user/magvasp/runs"
    )
    execution = ExecutionRequest(
        node="node02", step_id=step_id, run_id="evidence-run", mpi_ranks=2
    )
    observed_at = datetime.now(UTC) - timedelta(seconds=3)
    snapshot = _snapshot(observed_at)
    grant = build_resource_grant(
        bundle,
        profile,
        execution,
        snapshot,
        approver="cluster_user",
        confirm=RESOURCE_GRANT_CONFIRM,
        now=observed_at,
        valid_for=timedelta(minutes=15),
    )
    smoke = SmokeTestEvidence(
        run_id=execution.run_id,
        node=execution.node,
        mpi_ranks=execution.mpi_ranks,
        cpu_ids=grant.cpu_ids,
        hostname=f"{execution.node}.example.org",
        started_at=observed_at + timedelta(seconds=1),
        finished_at=observed_at + timedelta(seconds=2),
        exit_code=0,
        rank_cpu_ids=grant.cpu_ids,
        stdout_sha256="1" * 64,
        stderr_sha256="2" * 64,
        output_bytes=32,
        passed=True,
    )
    transfer_plan = build_transfer_plan(
        bundle,
        profile,
        TransferRequest(node=execution.node, run_id=execution.run_id),
    )
    transfer = transfer_report_model(
        {
            **transfer_plan,
            "mode": "transfer",
            "ready_to_transfer": False,
            "transferred": True,
            "remote_hashes_verified": True,
            "remote_directory_created": True,
        }
    )

    paths = {
        "profile": tmp_path / "profile.json",
        "request": tmp_path / "request.json",
        "snapshot": tmp_path / "snapshot.json",
        "grant": tmp_path / "grant.json",
        "smoke": tmp_path / "smoke.json",
        "transfer": tmp_path / "transfer.json",
    }
    for name, model in (
        ("profile", profile),
        ("request", execution),
        ("snapshot", snapshot),
        ("grant", grant),
        ("smoke", smoke),
        ("transfer", transfer),
    ):
        _write_model(paths[name], model)

    launch_request = LaunchRequest(
        backend=LaunchBackend.REMOTE_MPI,
        confirm=REMOTE_CONFIRM,
        output=str(tmp_path / "run-record"),
        run_id=execution.run_id,
        profile=str(paths["profile"]),
        bundle=str(bundle),
        request=str(paths["request"]),
        snapshot=str(paths["snapshot"]),
        resource_grant=str(paths["grant"]),
        smoke_report=str(paths["smoke"]),
        transfer_report=str(paths["transfer"]),
    )
    return EvidenceContext(
        bundle=bundle,
        profile=profile,
        execution=execution,
        snapshot=snapshot,
        grant=grant,
        smoke=smoke,
        transfer=transfer,
        review=review,
        launch_request=launch_request,
        paths=paths,
    )


def _remote_start_result(context: EvidenceContext) -> dict[str, object]:
    return {
        "marker": f"MAGVASP_REMOTE_VASP_V1 {context.execution.run_id}",
        "workdir": (
            f"{context.transfer.remote_directory}/{context.execution.step_id}"
        ),
        "remote_pid": 4242,
        "executable": "vasp_std",
    }


def _assert_blocked(
    context: EvidenceContext,
    monkeypatch: pytest.MonkeyPatch,
    expected_blocker: str,
) -> None:
    remote_start = Mock(side_effect=AssertionError("remote start must stay blocked"))
    monkeypatch.setattr(runtime, "start_remote_vasp", remote_start)
    record = runtime.launch(context.launch_request)
    assert record.state == JobState.BLOCKED
    assert expected_blocker in record.blockers
    remote_start.assert_not_called()


def test_complete_evidence_starts_remote_vasp_exactly_once(
    evidence_context: EvidenceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_start = Mock(return_value=_remote_start_result(evidence_context))
    monkeypatch.setattr(runtime, "start_remote_vasp", remote_start)

    record = runtime.launch(evidence_context.launch_request)

    assert record.state == JobState.RUNNING
    assert record.cluster_execution == "remote_mpi"
    assert record.remote_pid == 4242
    remote_start.assert_called_once()


def test_launch_probe_may_be_newer_than_bound_smoke(
    evidence_context: EvidenceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch-time re-probe must not invalidate an earlier approved smoke."""

    latest = _snapshot(datetime.now(UTC) - timedelta(seconds=1))
    assert latest.observed_at > evidence_context.smoke.finished_at
    _write_model(evidence_context.paths["snapshot"], latest)
    remote_start = Mock(return_value=_remote_start_result(evidence_context))
    monkeypatch.setattr(runtime, "start_remote_vasp", remote_start)

    record = runtime.launch(evidence_context.launch_request)

    assert record.state == JobState.RUNNING
    remote_start.assert_called_once()


@pytest.mark.parametrize("review_state", ["pending", "missing"])
def test_unaccepted_or_missing_review_blocks_launch(
    evidence_context: EvidenceContext,
    monkeypatch: pytest.MonkeyPatch,
    review_state: str,
) -> None:
    path = evidence_context.bundle / "input.review.json"
    if review_state == "missing":
        path.unlink()
    else:
        pending = InputReviewCard.model_validate(
            {
                **evidence_context.review.model_dump(mode="json"),
                "status": "pending",
                "awaiting_human": True,
                "decision": None,
                "reviewer": None,
                "recorded_at": None,
            }
        )
        _write_model(path, pending)

    _assert_blocked(
        evidence_context, monkeypatch, "INPUT_REVIEW_NOT_ACCEPTED"
    )


def test_missing_probe_blocks_launch(
    evidence_context: EvidenceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_context.launch_request = evidence_context.launch_request.model_copy(
        update={"snapshot": None}
    )
    _assert_blocked(evidence_context, monkeypatch, "PROBE_SNAPSHOT_REQUIRED")


@pytest.mark.parametrize(
    ("updates", "expected_blocker"),
    [
        ({"observed_at": datetime.now(UTC) - timedelta(minutes=2)}, "SNAPSHOT_STALE"),
        (
            {"requested_node": "node03", "hostname": "node03.example.org"},
            "PROBE_NODE_MISMATCH",
        ),
    ],
)
def test_stale_or_mismatched_probe_blocks_launch(
    evidence_context: EvidenceContext,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    expected_blocker: str,
) -> None:
    snapshot = ProbeSnapshot.model_validate(
        {**evidence_context.snapshot.model_dump(), **updates}
    )
    _write_model(evidence_context.paths["snapshot"], snapshot)
    _assert_blocked(evidence_context, monkeypatch, expected_blocker)


def test_expired_resource_grant_blocks_launch(
    evidence_context: EvidenceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    grant = ResourceGrant.model_validate(
        {
            **evidence_context.grant.model_dump(mode="json"),
            "valid_from": now - timedelta(minutes=20),
            "valid_until": now - timedelta(minutes=10),
            "approved_at": now - timedelta(minutes=15),
        }
    )
    _write_model(evidence_context.paths["grant"], grant)
    _assert_blocked(evidence_context, monkeypatch, "RESOURCE_GRANT_EXPIRED")


@pytest.mark.parametrize(
    "updates",
    [
        {"workflow_manifest_sha256": ZERO_HASH},
        {"step_manifest_sha256": ZERO_HASH},
        {"run_sha256": ZERO_HASH},
        {
            "input_sha256": {
                "POSCAR": ZERO_HASH,
                "POTCAR": ZERO_HASH,
                "INCAR": ZERO_HASH,
                "KPOINTS": ZERO_HASH,
            }
        },
    ],
    ids=["workflow", "step", "run", "inputs"],
)
def test_hash_mismatched_resource_grant_blocks_launch(
    evidence_context: EvidenceContext,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
) -> None:
    grant = ResourceGrant.model_validate(
        {**evidence_context.grant.model_dump(), **updates}
    )
    _write_model(evidence_context.paths["grant"], grant)
    _assert_blocked(evidence_context, monkeypatch, "RESOURCE_GRANT_MISMATCH")


def test_stale_smoke_evidence_blocks_launch(
    evidence_context: EvidenceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    smoke = SmokeTestEvidence.model_validate(
        {
            **evidence_context.smoke.model_dump(mode="json"),
            "started_at": now - timedelta(minutes=20, seconds=1),
            "finished_at": now - timedelta(minutes=20),
        }
    )
    _write_model(evidence_context.paths["smoke"], smoke)
    _assert_blocked(evidence_context, monkeypatch, "SMOKE_STALE")


def test_resource_mismatched_smoke_evidence_blocks_launch(
    evidence_context: EvidenceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = SmokeTestEvidence.model_validate(
        {
            **evidence_context.smoke.model_dump(),
            "node": "node03",
            "hostname": "node03.example.org",
        }
    )
    _write_model(evidence_context.paths["smoke"], smoke)
    _assert_blocked(
        evidence_context, monkeypatch, "RESOURCE_GRANT_SMOKE_MISMATCH"
    )


@pytest.mark.parametrize(
    ("updates", "case"),
    [
        ({"workflow_id": "other-workflow"}, "workflow"),
        ({"run_id": "other-run"}, "run"),
        ({"node": "node03"}, "node"),
        ({"remote_directory": "/home/cluster_user/magvasp/runs/other-run"}, "remote-dir"),
    ],
)
def test_transfer_identity_mismatch_blocks_launch(
    evidence_context: EvidenceContext,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    case: str,
) -> None:
    del case
    transfer = TransferReport.model_validate(
        {**evidence_context.transfer.model_dump(mode="json"), **updates}
    )
    _write_model(evidence_context.paths["transfer"], transfer)
    _assert_blocked(evidence_context, monkeypatch, "TRANSFER_EVIDENCE_MISMATCH")


@pytest.mark.parametrize("inventory_change", ["missing", "extra"])
def test_transfer_file_inventory_mismatch_blocks_launch(
    evidence_context: EvidenceContext,
    monkeypatch: pytest.MonkeyPatch,
    inventory_change: str,
) -> None:
    files = list(evidence_context.transfer.files)
    if inventory_change == "missing":
        files.pop()
    else:
        files.append(
            TransferFile(
                relative_path="unexpected.txt",
                size=1,
                sha256=ZERO_HASH,
            )
        )
    transfer = TransferReport.model_validate(
        {
            **evidence_context.transfer.model_dump(mode="json"),
            "files": [item.model_dump(mode="json") for item in files],
        }
    )
    _write_model(evidence_context.paths["transfer"], transfer)
    _assert_blocked(evidence_context, monkeypatch, "TRANSFER_EVIDENCE_MISMATCH")
