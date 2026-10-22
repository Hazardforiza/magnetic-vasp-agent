from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.execution_approval import (
    RESOURCE_GRANT_CONFIRM,
    build_resource_grant,
    issue_resource_grant,
)
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    Cpu,
    ExecutionRequest,
    ProbeSnapshot,
    ProcessObservation,
)
from magnetic_vasp_agent.input_review import build_review_card, record_input_review
from magnetic_vasp_agent.models import PotentialPolicy, PotentialRecord, PotentialRegistry
from magnetic_vasp_agent.workflow import digest, materialize_workflow, prepare_workflow
from magnetic_vasp_agent.workflow_models import Workflow
from magnetic_vasp_agent.mcif_workflow import plan_workflow_from_mcif

ROOT = Path(__file__).parents[1]


def _snapshot(*, node: str = "node02", source: str = "ssh_probe") -> ProbeSnapshot:
    cpus = [
        Cpu(cpu=i, core=i % 20, socket=(i % 20) // 10, online=True)
        for i in range(40)
    ]
    return ProbeSnapshot(
        source=source,
        observed_at=datetime.now(UTC),
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
def grant_context(tmp_path: Path):
    records = []
    for element in ("Fe", "O"):
        payload = f"Synthetic test potential: {element}\n".encode()
        path = tmp_path / "potentials" / element / "POTCAR"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append(
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
    request_path = tmp_path / "workflow.json"
    plan_workflow_from_mcif([source], request_path, workflow_id="grant-demo")
    prepared = prepare_workflow(
        Workflow.model_validate_json(request_path.read_bytes()),
        PotentialRegistry(
            root=str(tmp_path / "potentials"), family="PBE_54", records=records
        ),
        PotentialPolicy(family="PBE_54", preferred={}),
        tmp_path,
    )
    bundle = tmp_path / "bundle"
    materialize_workflow(prepared, bundle)
    review_log = tmp_path / "reviews.jsonl"
    build_review_card(bundle, log_path=review_log)
    record_input_review(
        bundle,
        reviewer="cluster_user",
        decision="accept",
        log_path=review_log,
    )
    step_id = json_step_ids(bundle)[0]
    profile = ClusterProfile(
        execution_enabled=True, remote_root="/home/cluster_user/magvasp/runs"
    )
    request = ExecutionRequest(
        node="node02", step_id=step_id, run_id="grant-run", mpi_ranks=2
    )
    snapshot = _snapshot()
    return bundle, profile, request, snapshot


def json_step_ids(bundle: Path) -> list[str]:
    import json

    manifest = json.loads((bundle / "workflow.manifest.json").read_bytes())
    return list(manifest["order"])


def test_resource_grant_revalidates_and_binds_all_artifacts(grant_context, tmp_path):
    bundle, profile, request, snapshot = grant_context
    approved_at = snapshot.observed_at
    grant = build_resource_grant(
        bundle,
        profile,
        request,
        snapshot,
        approver="cluster_user",
        confirm=RESOURCE_GRANT_CONFIRM,
        now=approved_at,
        valid_for=timedelta(minutes=15),
    )

    assert grant.workflow_id == "grant-demo"
    assert grant.node == "node02"
    assert grant.mpi_ranks == 2
    assert grant.cpu_ids == [0, 1]
    assert grant.approver == "cluster_user"
    assert set(grant.input_sha256) == {"POSCAR", "POTCAR", "INCAR", "KPOINTS"}
    assert grant.run_sha256 == digest(
        __import__("magnetic_vasp_agent.workflow", fromlist=["json_bytes"]).json_bytes(
            request.model_dump(mode="json")
        )
    )
    assert grant.covers(approved_at + timedelta(minutes=1))

    output = tmp_path / "resource-grant.json"
    written = issue_resource_grant(
        bundle,
        profile,
        request,
        snapshot,
        output,
        approver="cluster_user",
        confirm=RESOURCE_GRANT_CONFIRM,
        now=approved_at,
    )
    assert written.grant_id == written.model_validate_json(output.read_bytes()).grant_id
    with pytest.raises(FileExistsError):
        issue_resource_grant(
            bundle,
            profile,
            request,
            snapshot,
            output,
            approver="cluster_user",
            confirm=RESOURCE_GRANT_CONFIRM,
            now=approved_at,
        )


def test_resource_grant_requires_exact_confirmation_and_human_review(grant_context):
    bundle, profile, request, snapshot = grant_context
    with pytest.raises(ValueError, match="I_APPROVE_RESOURCE_GRANT"):
        build_resource_grant(
            bundle,
            profile,
            request,
            snapshot,
            approver="cluster_user",
            confirm="yes",
            now=snapshot.observed_at,
        )
    with pytest.raises(ValueError, match="named human"):
        build_resource_grant(
            bundle,
            profile,
            request,
            snapshot,
            approver="agent",
            confirm=RESOURCE_GRANT_CONFIRM,
            now=snapshot.observed_at,
        )


def test_resource_grant_rejects_stale_or_offline_snapshot(grant_context):
    bundle, profile, request, snapshot = grant_context
    stale = snapshot.model_copy(
        update={"observed_at": snapshot.observed_at - timedelta(seconds=61)}
    )
    with pytest.raises(ValueError, match="SNAPSHOT_STALE"):
        build_resource_grant(
            bundle,
            profile,
            request,
            stale,
            approver="cluster_user",
            confirm=RESOURCE_GRANT_CONFIRM,
            now=snapshot.observed_at,
        )
    offline = snapshot.model_copy(update={"source": "offline_fixture"})
    with pytest.raises(ValueError, match="OFFLINE_EVIDENCE"):
        build_resource_grant(
            bundle,
            profile,
            request,
            offline,
            approver="cluster_user",
            confirm=RESOURCE_GRANT_CONFIRM,
            now=snapshot.observed_at,
        )


def test_resource_grant_cli_writes_json(grant_context, tmp_path):
    bundle, profile, request, snapshot = grant_context
    profile_path = tmp_path / "profile.json"
    request_path = tmp_path / "request.json"
    snapshot_path = tmp_path / "snapshot.json"
    profile_path.write_bytes(profile.model_dump_json().encode())
    request_path.write_bytes(request.model_dump_json().encode())
    snapshot_path.write_bytes(snapshot.model_dump_json().encode())
    output = tmp_path / "grant.json"
    result = CliRunner().invoke(
        app,
        [
            "execution",
            "grant",
            "--bundle",
            str(bundle),
            "--profile",
            str(profile_path),
            "--request",
            str(request_path),
            "--snapshot",
            str(snapshot_path),
            "--approver",
            "cluster_user",
            "--confirm",
            RESOURCE_GRANT_CONFIRM,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert '"approver": "cluster_user"' in output.read_text(encoding="utf-8")
