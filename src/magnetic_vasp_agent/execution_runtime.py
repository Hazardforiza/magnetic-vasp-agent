"""Gated launch / status / cancel.

Local launches reuse the fixed-program simulator. Remote MPI starts only after
an enabled cluster profile and a complete, fresh, mutually bound evidence set
are supplied together with the exact confirmation phrase. Cancel still refuses
remote PIDs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from magnetic_vasp_agent.execution import _read_step, write_report
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    ExecutionRequest,
    ExecutionStage,
    ExecutionStateMachine,
    ProbeSnapshot,
    ResourceGrant,
    SmokeTestEvidence,
)
from magnetic_vasp_agent.execution_remote import RemoteExecutionError, start_remote_vasp
from magnetic_vasp_agent.execution_artifacts import (
    ArtifactReleaseEvidence,
    release_matches_target,
)
from magnetic_vasp_agent.execution_runtime_models import (
    LOCAL_CONFIRM,
    REMOTE_CONFIRM,
    JobState,
    LaunchBackend,
    LaunchRequest,
    RunRecord,
)
from magnetic_vasp_agent.execution_simulator import LocalSimulation
from magnetic_vasp_agent.input_review_models import InputReviewCard
from magnetic_vasp_agent.transfer import bundle_transfer_hashes, transfer_report_model
from magnetic_vasp_agent.workflow import INPUT_NAMES, digest, json_bytes

RECORD_NAME = "run.record.json"
COMMAND_MARKER = "SIMULATION ONLY"
MAX_SMOKE_AGE_SECONDS = 15 * 60


class ExecutionRuntimeError(ValueError):
    """Raised when a launch/status/cancel request is refused."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def execution_request_sha256(request: ExecutionRequest) -> str:
    """Digest the normalized, typed execution request used by a resource grant."""

    return _digest(json_bytes(request.model_dump(mode="json")))


def _stamp(record: RunRecord) -> RunRecord:
    payload = record.model_dump(mode="json")
    payload.pop("record_sha256", None)
    return record.model_copy(update={"record_sha256": _digest(json_bytes(payload))})


def _write_record(directory: Path, record: RunRecord) -> RunRecord:
    stamped = _stamp(record)
    write_report(stamped.model_dump(mode="json"), directory / RECORD_NAME)
    return stamped


def _replace_record(directory: Path, record: RunRecord) -> RunRecord:
    stamped = _stamp(record)
    path = directory / RECORD_NAME
    path.write_bytes(json_bytes(stamped.model_dump(mode="json")))
    return stamped


def read_run(directory: Path) -> RunRecord:
    path = directory / RECORD_NAME
    if not path.is_file() or path.is_symlink():
        raise ExecutionRuntimeError(f"no local run record at {path}")
    return RunRecord.model_validate_json(path.read_bytes())


def _blocked_remote(output: Path, run_id: str, blockers: list[str]) -> RunRecord:
    output.mkdir(parents=True, exist_ok=True)
    record = RunRecord(
        backend=LaunchBackend.REMOTE_MPI,
        state=JobState.BLOCKED,
        run_id=run_id,
        output_dir=str(output.resolve()),
        events=["blocked"],
        blockers=blockers,
        owned_locally=False,
        cluster_execution="disabled",
    )
    return (
        _replace_record(output, record)
        if (output / RECORD_NAME).is_file()
        else _write_record(output, record)
    )


def _regular_bytes(path_text: str, *, label: str) -> bytes:
    path = Path(path_text)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular local file")
    return path.read_bytes()


def _accepted_review(
    bundle: Path,
    *,
    workflow_id: str,
    step_id: str,
    input_sha256: dict[str, str],
) -> bool:
    path = bundle / "input.review.json"
    if path.is_symlink() or not path.is_file():
        return False
    card = InputReviewCard.model_validate_json(path.read_bytes())
    matching_steps = [item for item in card.steps if item.step_id == step_id]
    return (
        card.workflow_id == workflow_id
        and Path(card.bundle).resolve() == bundle.resolve()
        and card.integrity_valid
        and card.status == "reviewed"
        and card.decision == "accept"
        and not card.awaiting_human
        and bool(card.reviewer and card.recorded_at)
        and len(matching_steps) == 1
        and matching_steps[0].file_hashes == input_sha256
    )


def _verified_transfer(
    transfer,
    *,
    profile: ClusterProfile,
    bundle: Path,
    workflow_id: str,
    step_id: str,
    run_id: str,
    node: str,
    workflow_manifest_sha256: str,
    step_manifest_sha256: str,
    input_sha256: dict[str, str],
) -> bool:
    if profile.remote_root is None:
        return False
    files = {item.relative_path: item.sha256 for item in transfer.files}
    if len(files) != len(transfer.files):
        return False
    expected = bundle_transfer_hashes(bundle)
    # Keep these explicit checks close to the launch identity even though the
    # complete inventory below also contains them.
    if expected.get("workflow.manifest.json") != workflow_manifest_sha256:
        return False
    if expected.get(f"{step_id}/manifest.json") != step_manifest_sha256:
        return False
    if any(expected.get(f"{step_id}/{name}") != input_sha256[name] for name in INPUT_NAMES):
        return False
    expected_remote = f"{profile.remote_root}/{workflow_id}--{run_id}"
    return (
        transfer.mode == "transfer"
        and transfer.workflow_id == workflow_id
        and transfer.run_id == run_id
        and transfer.node == node
        and Path(transfer.local_bundle).resolve() == bundle.resolve()
        and transfer.remote_directory == expected_remote
        and transfer.transferred
        and transfer.remote_hashes_verified
        and transfer.remote_directory_created
        and transfer.execution_enabled
        and not transfer.ready_to_transfer
        and not transfer.blockers
        and files == expected
    )


def _launch_remote(request: LaunchRequest, output: Path) -> RunRecord:
    missing = [
        name
        for name in (
            "profile",
            "bundle",
            "request",
            "transfer_report",
            "smoke_report",
            "resource_grant",
        )
        if not getattr(request, name)
    ]
    if missing:
        return _blocked_remote(
            output,
            request.run_id,
            [
                "REMOTE_EXECUTION_DISABLED" if "profile" in missing else "LAUNCH_EVIDENCE_MISSING",
                "SSH_KEY_LOGIN_UNVERIFIED",
                "MPI_SMOKE_TEST_NOT_PASSED",
            ],
        )
    try:
        profile = ClusterProfile.model_validate_json(
            _regular_bytes(request.profile, label="profile")
        )
    except (ValueError, OSError):
        return _blocked_remote(output, request.run_id, ["CLUSTER_PROFILE_INVALID"])
    if not profile.execution_enabled:
        return _blocked_remote(output, request.run_id, ["REMOTE_EXECUTION_DISABLED"])
    try:
        execution = ExecutionRequest.model_validate_json(
            _regular_bytes(request.request, label="execution request")
        )
    except (ValueError, OSError):
        return _blocked_remote(output, request.run_id, ["EXECUTION_REQUEST_INVALID"])
    if request.run_id != execution.run_id:
        return _blocked_remote(output, request.run_id, ["RUN_ID_MISMATCH"])
    try:
        grant = ResourceGrant.model_validate_json(
            _regular_bytes(request.resource_grant, label="resource grant")
        )
    except (ValueError, OSError):
        return _blocked_remote(output, request.run_id, ["RESOURCE_GRANT_INVALID"])
    bundle = Path(request.bundle).resolve()
    try:
        manifest, step, incar = _read_step(bundle, execution.step_id)
    except (ValueError, OSError, KeyError, TypeError, IndexError, AttributeError):
        return _blocked_remote(output, request.run_id, ["INPUT_BUNDLE_INVALID"])
    workflow_id = manifest["workflow_id"]
    if grant.workflow_id != workflow_id:
        return _blocked_remote(output, request.run_id, ["RESOURCE_GRANT_WORKFLOW_MISMATCH"])
    input_sha256 = {name: step["files"][name] for name in INPUT_NAMES}
    workflow_manifest_sha256 = digest((bundle / "workflow.manifest.json").read_bytes())
    step_manifest_sha256 = manifest["steps"][execution.step_id]["manifest_sha256"]
    run_sha256 = execution_request_sha256(execution)
    try:
        review_accepted = _accepted_review(
            bundle,
            workflow_id=workflow_id,
            step_id=execution.step_id,
            input_sha256=input_sha256,
        )
    except (ValueError, OSError):
        return _blocked_remote(output, request.run_id, ["INPUT_REVIEW_INVALID"])
    if not review_accepted:
        return _blocked_remote(output, request.run_id, ["INPUT_REVIEW_NOT_ACCEPTED"])
    if not request.snapshot:
        return _blocked_remote(output, request.run_id, ["PROBE_SNAPSHOT_REQUIRED"])
    try:
        snapshot = ProbeSnapshot.model_validate_json(
            _regular_bytes(request.snapshot, label="probe snapshot")
        )
    except (ValueError, OSError):
        return _blocked_remote(output, request.run_id, ["PROBE_SNAPSHOT_INVALID"])
    try:
        smoke = SmokeTestEvidence.model_validate_json(
            _regular_bytes(request.smoke_report, label="smoke report")
        )
    except (ValueError, OSError):
        return _blocked_remote(output, request.run_id, ["SMOKE_EVIDENCE_INVALID"])
    try:
        transfer_bytes = _regular_bytes(request.transfer_report, label="transfer report")
        transfer = transfer_report_model(json.loads(transfer_bytes))
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
        return _blocked_remote(output, request.run_id, ["TRANSFER_REPORT_INVALID"])

    now = datetime.now(UTC)
    gate = ExecutionStateMachine(
        workflow_id,
        execution.step_id,
        execution.run_id,
        workflow_manifest_sha256=workflow_manifest_sha256,
        step_manifest_sha256=step_manifest_sha256,
        run_sha256=run_sha256,
        input_sha256=input_sha256,
    )
    try:
        state = gate.record_resource_grant(grant, execution, now=now)
    except ValueError:
        return _blocked_remote(output, request.run_id, ["RESOURCE_GRANT_MISMATCH"])
    if state.stage == ExecutionStage.BLOCKED:
        return _blocked_remote(output, request.run_id, state.blockers)
    if not grant.binds_artifacts(
        workflow_id=workflow_id,
        step_id=execution.step_id,
        run_id=execution.run_id,
        workflow_manifest_sha256=workflow_manifest_sha256,
        step_manifest_sha256=step_manifest_sha256,
        run_sha256=run_sha256,
        input_sha256=input_sha256,
    ):
        return _blocked_remote(output, request.run_id, ["RESOURCE_GRANT_ARTIFACT_MISMATCH"])
    if snapshot.source != "ssh_probe":
        return _blocked_remote(output, request.run_id, ["OFFLINE_EVIDENCE"])
    try:
        state = gate.record_probe(snapshot, execution, now=now, profile=profile)
    except ValueError:
        return _blocked_remote(output, request.run_id, ["PROBE_SNAPSHOT_MISMATCH"])
    if state.stage == ExecutionStage.BLOCKED:
        return _blocked_remote(output, request.run_id, state.blockers)
    if (
        not snapshot.module_loaded
        or snapshot.mpi_path != profile.mpi_path
        or not snapshot.executables.get(incar.executable, False)
    ):
        return _blocked_remote(output, request.run_id, ["RUNTIME_MISMATCH"])
    smoke_age = (now - smoke.finished_at).total_seconds()
    if smoke_age < -5 or smoke_age > MAX_SMOKE_AGE_SECONDS:
        return _blocked_remote(output, request.run_id, ["SMOKE_STALE"])
    # The launch probe is deliberately collected after the separate smoke and
    # transfer approvals.  It proves current launch-time state; the smoke is
    # independently bound to the same grant and must fit inside that grant.
    if (
        smoke.started_at < grant.valid_from
        or smoke.finished_at > grant.valid_until
    ):
        return _blocked_remote(output, request.run_id, ["SMOKE_TIME_MISMATCH"])
    try:
        state = gate.record_smoke_test(smoke)
    except ValueError:
        return _blocked_remote(output, request.run_id, ["RESOURCE_GRANT_SMOKE_MISMATCH"])
    if state.stage == ExecutionStage.BLOCKED:
        return _blocked_remote(output, request.run_id, state.blockers)
    if not _verified_transfer(
        transfer,
        profile=profile,
        bundle=bundle,
        workflow_id=workflow_id,
        step_id=execution.step_id,
        run_id=execution.run_id,
        node=execution.node,
        workflow_manifest_sha256=workflow_manifest_sha256,
        step_manifest_sha256=step_manifest_sha256,
        input_sha256=input_sha256,
    ):
        return _blocked_remote(output, request.run_id, ["TRANSFER_EVIDENCE_MISMATCH"])

    dependencies = step["status"].get("dependencies") or []
    release = None
    if dependencies:
        if len(dependencies) != 1 or dependencies[0].get("artifact") != "CHGCAR":
            return _blocked_remote(output, request.run_id, ["UPSTREAM_ARTIFACT_UNSUPPORTED"])
        if not request.artifact_release:
            return _blocked_remote(output, request.run_id, ["UPSTREAM_ARTIFACT_RELEASE_REQUIRED"])
        try:
            release = ArtifactReleaseEvidence.model_validate_json(
                _regular_bytes(request.artifact_release, label="artifact release")
            )
        except (ValueError, OSError):
            return _blocked_remote(output, request.run_id, ["UPSTREAM_ARTIFACT_RELEASE_INVALID"])
        expected_target_remote = f"{transfer.remote_directory}/{execution.step_id}"
        if not release_matches_target(
            release,
            bundle=bundle,
            workflow_id=workflow_id,
            target_step=execution.step_id,
            target_run_id=execution.run_id,
            node=execution.node,
            target_remote_directory=expected_target_remote,
            target_transfer_report_sha256=digest(transfer_bytes),
        ):
            return _blocked_remote(output, request.run_id, ["UPSTREAM_ARTIFACT_RELEASE_MISMATCH"])

    cpu_ids = grant.cpu_ids
    _write_record(
        output,
        RunRecord(
            backend=LaunchBackend.REMOTE_MPI,
            state=JobState.PREPARED,
            run_id=execution.run_id,
            output_dir=str(output.resolve()),
            events=["prepared"],
            blockers=[],
            owned_locally=False,
            cluster_execution="disabled",
            node=execution.node,
            remote_directory=f"{transfer.remote_directory}/{execution.step_id}",
            executable=incar.executable,
        ),
    )
    try:
        started = start_remote_vasp(
            profile,
            execution,
            transfer,
            executable=incar.executable,
            cpu_ids=cpu_ids,
            artifact_sha256=release.artifact_sha256 if release else None,
            artifact_size=release.artifact_size if release else None,
        )
    except (RemoteExecutionError, ValueError, OSError) as exc:
        return _blocked_remote(output, request.run_id, [str(exc).split(":", 1)[0]])
    return _replace_record(
        output,
        RunRecord(
            backend=LaunchBackend.REMOTE_MPI,
            state=JobState.RUNNING,
            run_id=execution.run_id,
            output_dir=str(output.resolve()),
            events=["prepared", "running"],
            blockers=[],
            owned_locally=False,
            cluster_execution="remote_mpi",
            command_marker=started["marker"],
            node=execution.node,
            remote_directory=started["workdir"],
            remote_pid=started["remote_pid"],
            executable=started["executable"],
        ),
    )


def launch(request: LaunchRequest) -> RunRecord:
    """Start a local simulation, or a gated remote MPI/VASP run."""

    request = LaunchRequest.model_validate_json(request.model_dump_json())
    output = Path(request.output)
    if request.backend == LaunchBackend.REMOTE_MPI:
        if request.confirm != REMOTE_CONFIRM:
            raise ExecutionRuntimeError(
                "remote launch requires the exact phrase I_CONFIRM_REMOTE_LAUNCH"
            )
        if output.exists() or output.is_symlink():
            raise ExecutionRuntimeError(
                "remote launch output must be a new path before any cluster action"
            )
        return _launch_remote(request, output)

    if request.confirm != LOCAL_CONFIRM:
        raise ExecutionRuntimeError(
            "local launch requires the exact phrase I_CONFIRM_LOCAL_LAUNCH"
        )
    simulation = LocalSimulation(output, request.scenario, request.timeout_seconds)
    try:
        simulation.start()
        if request.scenario == "cancel":
            simulation.cancel()
        while simulation.poll() == "running":
            time.sleep(0.02)
        report = simulation.report()
        return _write_record(
            output,
            RunRecord(
                backend=LaunchBackend.LOCAL_SIMULATION,
                state=JobState(report["state"]),
                run_id=request.run_id,
                output_dir=str(output.resolve()),
                pid=simulation.process.pid if simulation.process else None,
                exit_code=report["exit_code"],
                events=list(report["events"]),
                owned_locally=True,
                command_marker=COMMAND_MARKER,
            ),
        )
    finally:
        simulation.close()


def status(directory: Path) -> RunRecord:
    """Re-read the local record. Does not contact the cluster."""

    return read_run(directory)


def _process_command(pid: int) -> str | None:
    if pid <= 0:
        return None
    if sys.platform == "win32":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        text = (result.stdout or "").strip()
        return text or None
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.is_file():
        return None
    return cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")


def cancel(directory: Path) -> RunRecord:
    """Cancel only a locally owned simulation whose live command still matches."""

    record = read_run(directory)
    if record.backend != LaunchBackend.LOCAL_SIMULATION or not record.owned_locally:
        raise ExecutionRuntimeError("cancel is refused for remote or unowned runs")
    if record.state != JobState.RUNNING:
        return record
    if record.pid is None:
        raise ExecutionRuntimeError("run record has no owned pid")
    command = _process_command(record.pid)
    if command is None:
        updated = record.model_copy(
            update={"state": JobState.CANCELLED, "events": [*record.events, "cancelled"]}
        )
        return _replace_record(directory, updated)
    if COMMAND_MARKER not in command:
        raise ExecutionRuntimeError(
            "refusing to signal a process whose command no longer matches the simulation marker"
        )
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(record.pid), "/T"],
            capture_output=True,
            timeout=8,
            check=False,
        )
    else:
        subprocess.run(["kill", "-TERM", str(record.pid)], check=False, timeout=8)
    updated = record.model_copy(
        update={"state": JobState.CANCELLED, "events": [*record.events, "cancelled"]}
    )
    return _replace_record(directory, updated)
