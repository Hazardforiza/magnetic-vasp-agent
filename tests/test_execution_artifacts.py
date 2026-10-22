from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from magnetic_vasp_agent import execution_runtime as runtime
from magnetic_vasp_agent.execution_approval import (
    RESOURCE_GRANT_CONFIRM,
    build_resource_grant,
)
from magnetic_vasp_agent.execution_artifacts import (
    ARTIFACT_RELEASE_CONFIRM,
    ArtifactReleaseError,
    artifact_release_valid,
    release_chgcar,
    release_matches_target,
)
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    Cpu,
    ExecutionRequest,
    ProbeSnapshot,
    SmokeTestEvidence,
)
from magnetic_vasp_agent.execution_runtime import _stamp
from magnetic_vasp_agent.execution_runtime_models import (
    REMOTE_CONFIRM,
    JobState,
    LaunchBackend,
    RunRecord,
    LaunchRequest,
)
from magnetic_vasp_agent.input_review import build_review_card, record_input_review
from magnetic_vasp_agent.models import PotentialPolicy, PotentialRecord, PotentialRegistry
from magnetic_vasp_agent.transfer import build_transfer_plan, transfer_report_model
from magnetic_vasp_agent.transfer_models import TransferRequest
from magnetic_vasp_agent.workflow import digest, json_bytes, materialize_workflow, prepare_workflow
from magnetic_vasp_agent.workflow_models import Workflow

ROOT = Path(__file__).parents[1]


@pytest.fixture
def release_context(tmp_path: Path) -> dict[str, object]:
    records = []
    for element in ("Mn", "Al"):
        payload = f"SYNTHETIC ONLY: {element}\n".encode()
        path = tmp_path / "potentials" / element / "POTCAR"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
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
    request_path = ROOT / "config/workflows/Mn3Al.scf-dos-band.json"
    prepared = prepare_workflow(
        Workflow.model_validate_json(request_path.read_bytes()),
        PotentialRegistry(
            root=str(tmp_path / "potentials"), family="PBE_54", records=records
        ),
        PotentialPolicy(family="PBE_54", preferred={}),
        request_path.parent,
    )
    bundle = tmp_path / "bundle"
    materialize_workflow(prepared, bundle)
    build_review_card(bundle, log_path=tmp_path / "reviews.jsonl")
    record_input_review(
        bundle,
        reviewer="cluster_user",
        decision="accept",
        log_path=tmp_path / "reviews.jsonl",
    )
    workflow = json.loads((bundle / "workflow.manifest.json").read_bytes())

    profile = ClusterProfile(
        execution_enabled=True, remote_root="/home/cluster_user/magvasp/runs"
    )
    source_run = _stamp(
        RunRecord(
            backend=LaunchBackend.REMOTE_MPI,
            state=JobState.RUNNING,
            run_id="source01",
            output_dir=str(tmp_path / "source-record"),
            events=["prepared", "running"],
            owned_locally=False,
            cluster_execution="remote_mpi",
            node="node03",
            remote_directory=(
                "/home/cluster_user/magvasp/runs/Mn3Al-scf-dos-band--source01/scf"
            ),
            remote_pid=4321,
            executable="vasp_std",
        )
    )
    status = {
        "schema_version": 1,
        "mode": "remote_status",
        "run_id": "source01",
        "node": "node03",
        "remote_directory": source_run.remote_directory,
        "remote_pid": 4321,
        "state": "completed",
        "pid_alive": False,
        "pid_cwd_matches": False,
        "pid_cwd_observation": "unknown",
        "files": {},
        "tails": {},
        "scientific_converged": None,
        "blockers": [],
        "limitations": [],
    }
    status_path = tmp_path / "status.json"
    status_path.write_bytes(json_bytes(status))

    fetched = tmp_path / "fetched"
    fetched.mkdir()
    fixture = ROOT / "tests/fixtures/postprocess/collinear"
    for name in ("OUTCAR", "OSZICAR", "vasprun.xml"):
        shutil.copyfile(fixture / name, fetched / name)
    shutil.copyfile(bundle / "scf/POSCAR", fetched / "POSCAR")
    shutil.copyfile(bundle / "scf/manifest.json", fetched / "manifest.json")
    (fetched / "CHGCAR").write_bytes(b"verified charge density\n")
    file_evidence = {
        path.name: {
            "size": path.stat().st_size,
            "sha256": digest(path.read_bytes()),
        }
        for path in fetched.iterdir()
    }
    fetch = {
        "schema_version": 1,
        "mode": "remote_fetch",
        "run_id": "source01",
        "node": "node03",
        "remote_directory": source_run.remote_directory,
        "destination": str(fetched.resolve()),
        "files": file_evidence,
        "total_bytes": sum(item["size"] for item in file_evidence.values()),
        "remote_hashes_verified": True,
        "local_hashes_verified": True,
        "potcar_fetched": False,
        "wavecar_fetched": False,
    }
    fetch_path = tmp_path / "fetch.json"
    fetch_path.write_bytes(json_bytes(fetch))

    transfer_plan = build_transfer_plan(
        bundle, profile, TransferRequest(node="node03", run_id="target01")
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
    transfer_path = tmp_path / "transfer.json"
    transfer_path.write_bytes(json_bytes(transfer.model_dump(mode="json")))
    return {
        "bundle": bundle,
        "workflow": workflow,
        "profile": profile,
        "source_run": source_run,
        "status": status,
        "status_path": status_path,
        "fetch": fetch,
        "fetch_path": fetch_path,
        "fetched": fetched,
        "transfer": transfer,
        "transfer_path": transfer_path,
    }


def _success_runner(context: dict[str, object], calls: list[tuple]) -> object:
    fetched = context["fetch"]
    assert isinstance(fetched, dict)
    item = fetched["files"]["CHGCAR"]

    def runner(argv: list[str], stdin: bytes, timeout: int) -> object:
        calls.append((argv, stdin, timeout))
        payload = (
            "MAGVASP_ARTIFACT_RELEASE_V1\n"
            f"artifact=CHGCAR|{item['size']}|{item['sha256']}\n"
            "MAGVASP_ARTIFACT_RELEASE_END\n"
        ).encode("ascii")
        return SimpleNamespace(returncode=0, stdout=payload, stderr=b"")

    return runner


def _release(context: dict[str, object], *, runner=None):
    return release_chgcar(
        bundle=context["bundle"],
        profile=context["profile"],
        source_run=context["source_run"],
        source_status=context["status_path"],
        fetch_report=context["fetch_path"],
        target_transfer_report=context["transfer_path"],
        target_step="dos",
        confirm=ARTIFACT_RELEASE_CONFIRM,
        runner=runner,
        now=datetime(2026, 9, 22, tzinfo=UTC),
    )


def test_release_binds_converged_fetch_to_declared_dependency(
    release_context: dict[str, object],
) -> None:
    calls: list[tuple] = []
    evidence = _release(
        release_context, runner=_success_runner(release_context, calls)
    )

    assert len(calls) == 1
    argv, stdin, timeout = calls[0]
    assert argv[0] == "ssh"
    assert timeout == 15 * 60
    script = stdin.decode("ascii")
    assert "CHGCAR" in script
    assert "OUTCAR" in script
    assert "cp --" in script and "ln --" in script
    assert evidence.source_step == "scf"
    assert evidence.target_step == "dos"
    assert evidence.source_converged is True
    assert artifact_release_valid(evidence)
    assert release_matches_target(
        evidence,
        bundle=release_context["bundle"],
        workflow_id="Mn3Al-scf-dos-band",
        target_step="dos",
        target_run_id="target01",
        node="node03",
        target_remote_directory=(
            "/home/cluster_user/magvasp/runs/Mn3Al-scf-dos-band--target01/dos"
        ),
        target_transfer_report_sha256=digest(
            release_context["transfer_path"].read_bytes()
        ),
    )


def test_release_uses_a_timezone_aware_timestamp_by_default(
    release_context: dict[str, object],
) -> None:
    evidence = release_chgcar(
        bundle=release_context["bundle"],
        profile=release_context["profile"],
        source_run=release_context["source_run"],
        source_status=release_context["status_path"],
        fetch_report=release_context["fetch_path"],
        target_transfer_report=release_context["transfer_path"],
        target_step="dos",
        confirm=ARTIFACT_RELEASE_CONFIRM,
        runner=_success_runner(release_context, []),
    )

    assert evidence.promoted_at.utcoffset() is not None


def test_release_refuses_wrong_confirmation_without_remote_call(
    release_context: dict[str, object],
) -> None:
    with pytest.raises(ArtifactReleaseError, match="exact phrase"):
        release_chgcar(
            bundle=release_context["bundle"],
            profile=release_context["profile"],
            source_run=release_context["source_run"],
            source_status=release_context["status_path"],
            fetch_report=release_context["fetch_path"],
            target_transfer_report=release_context["transfer_path"],
            target_step="dos",
            confirm="yes",
            runner=lambda *args, **kwargs: pytest.fail("must not contact SSH"),
        )


def test_release_refuses_running_source(
    release_context: dict[str, object],
) -> None:
    status = dict(release_context["status"])
    status.update(state="running", pid_alive=True, pid_cwd_observation="yes")
    release_context["status_path"].write_bytes(json_bytes(status))
    with pytest.raises(ArtifactReleaseError, match="not cleanly completed"):
        _release(
            release_context,
            runner=lambda *args, **kwargs: pytest.fail("must not contact SSH"),
        )


def test_release_refuses_changed_local_chgcar(
    release_context: dict[str, object],
) -> None:
    (release_context["fetched"] / "CHGCAR").write_bytes(b"changed\n")
    with pytest.raises(ArtifactReleaseError, match="hash/size changed"):
        _release(
            release_context,
            runner=lambda *args, **kwargs: pytest.fail("must not contact SSH"),
        )


def test_release_refuses_transfer_missing_an_unrelated_step_file(
    release_context: dict[str, object],
) -> None:
    transfer = release_context["transfer"]
    files = [
        item
        for item in transfer.files
        if item.relative_path != "band/KPOINTS"
    ]
    altered = transfer.model_copy(update={"files": files})
    release_context["transfer_path"].write_bytes(
        json_bytes(altered.model_dump(mode="json"))
    )
    with pytest.raises(ArtifactReleaseError, match="INVENTORY_MISMATCH"):
        _release(
            release_context,
            runner=lambda *args, **kwargs: pytest.fail("must not contact SSH"),
        )


def test_tampered_release_report_is_not_launch_evidence(
    release_context: dict[str, object],
) -> None:
    evidence = _release(
        release_context, runner=_success_runner(release_context, [])
    )
    tampered = evidence.model_copy(update={"artifact_sha256": "0" * 64})
    assert not artifact_release_valid(tampered)
    assert not release_matches_target(
        tampered,
        bundle=release_context["bundle"],
        workflow_id="Mn3Al-scf-dos-band",
        target_step="dos",
        target_run_id="target01",
        node="node03",
        target_remote_directory=(
            "/home/cluster_user/magvasp/runs/Mn3Al-scf-dos-band--target01/dos"
        ),
        target_transfer_report_sha256=digest(
            release_context["transfer_path"].read_bytes()
        ),
    )


def test_dependent_launch_requires_and_forwards_release_evidence(
    release_context: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = _release(
        release_context, runner=_success_runner(release_context, [])
    )
    release_path = tmp_path / "release.json"
    release_path.write_bytes(json_bytes(release.model_dump(mode="json")))

    observed = datetime.now(UTC) - timedelta(seconds=3)
    snapshot = ProbeSnapshot(
        source="ssh_probe",
        observed_at=observed,
        requested_node="node03",
        hostname="node03.example.org",
        username="cluster_user",
        cpus=[
            Cpu(cpu=index, core=index % 20, socket=(index % 20) // 10, online=True)
            for index in range(40)
        ],
        allowed_cpu_ids=list(range(40)),
        processes=[],
        load1=0.0,
        mem_available_kib=8 * 1024 * 1024,
        module_loaded=True,
        mpi_path="/opt/intel/oneapi/mpi/bin/mpirun",
        executables={"vasp_std": True, "vasp_ncl": True, "vasp_gam": True},
    )
    request = ExecutionRequest(
        node="node03", step_id="dos", run_id="target01", mpi_ranks=2
    )
    grant = build_resource_grant(
        release_context["bundle"],
        release_context["profile"],
        request,
        snapshot,
        approver="cluster_user",
        confirm=RESOURCE_GRANT_CONFIRM,
        now=observed,
        valid_for=timedelta(minutes=15),
    )
    smoke = SmokeTestEvidence(
        run_id="target01",
        node="node03",
        mpi_ranks=2,
        cpu_ids=grant.cpu_ids,
        hostname="node03.example.org",
        started_at=observed + timedelta(seconds=1),
        finished_at=observed + timedelta(seconds=2),
        exit_code=0,
        rank_cpu_ids=grant.cpu_ids,
        stdout_sha256="1" * 64,
        stderr_sha256="2" * 64,
        output_bytes=16,
        passed=True,
    )
    paths = {
        "profile": tmp_path / "profile.json",
        "request": tmp_path / "request.json",
        "snapshot": tmp_path / "snapshot.json",
        "grant": tmp_path / "grant.json",
        "smoke": tmp_path / "smoke.json",
    }
    for name, model in (
        ("profile", release_context["profile"]),
        ("request", request),
        ("snapshot", snapshot),
        ("grant", grant),
        ("smoke", smoke),
    ):
        paths[name].write_bytes(json_bytes(model.model_dump(mode="json")))

    calls: list[dict[str, object]] = []

    def start_remote(*args, **kwargs):
        calls.append(kwargs)
        return {
            "marker": "MAGVASP_REMOTE_VASP_V1 target01",
            "workdir": (
                "/home/cluster_user/magvasp/runs/Mn3Al-scf-dos-band--target01/dos"
            ),
            "remote_pid": 5432,
            "executable": "vasp_std",
        }

    monkeypatch.setattr(runtime, "start_remote_vasp", start_remote)
    launch = LaunchRequest(
        backend=LaunchBackend.REMOTE_MPI,
        confirm=REMOTE_CONFIRM,
        output=str(tmp_path / "launch-record"),
        run_id="target01",
        profile=str(paths["profile"]),
        bundle=str(release_context["bundle"]),
        request=str(paths["request"]),
        snapshot=str(paths["snapshot"]),
        resource_grant=str(paths["grant"]),
        smoke_report=str(paths["smoke"]),
        transfer_report=str(release_context["transfer_path"]),
        artifact_release=str(release_path),
    )
    record = runtime.launch(launch)
    assert record.state == JobState.RUNNING
    assert len(calls) == 1
    assert calls[0]["artifact_sha256"] == release.artifact_sha256
    assert calls[0]["artifact_size"] == release.artifact_size

    missing = launch.model_copy(
        update={
            "output": str(tmp_path / "missing-release-record"),
            "artifact_release": None,
        }
    )
    blocked = runtime.launch(missing)
    assert blocked.state == JobState.BLOCKED
    assert "UPSTREAM_ARTIFACT_RELEASE_REQUIRED" in blocked.blockers
