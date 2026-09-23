"""Promote one verified upstream CHGCAR into a dependent remote step.

The operation is deliberately narrow: paths come from validated run/transfer
records, the only supported artifact is CHGCAR, and the remote program is a
fixed shell protocol.  No caller-supplied command or remote path is accepted.
"""

from __future__ import annotations

import hmac
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from magnetic_vasp_agent.execution import _read_step
from magnetic_vasp_agent.execution_models import ClusterProfile, ExecutionModel, SHA256
from magnetic_vasp_agent.execution_monitor import (
    FETCH_ALLOWED_FILES,
    MAX_COMMAND_OUTPUT_BYTES,
    _directory_guard,
    _execute,
    _fixed_script,
    _record_hash_valid,
    _regular_file_digest,
    _safe_remote_path,
    _ssh_argv,
    _validated_context,
)
from magnetic_vasp_agent.execution_runtime_models import RunRecord
from magnetic_vasp_agent.postprocess import analyze_calculation
from magnetic_vasp_agent.transfer import bundle_transfer_hashes, transfer_report_model
from magnetic_vasp_agent.workflow import digest, json_bytes
from magnetic_vasp_agent.workflow_models import safe_id

ARTIFACT_RELEASE_CONFIRM = "I_CONFIRM_ARTIFACT_RELEASE"
REQUIRED_FETCH_FILES = frozenset(
    {"CHGCAR", "OUTCAR", "OSZICAR", "vasprun.xml", "POSCAR", "manifest.json"}
)
RELEASE_TIMEOUT_SECONDS = 15 * 60


class ArtifactReleaseError(ValueError):
    """Raised when upstream evidence or the fixed promotion protocol fails."""


class FetchedFile(ExecutionModel):
    size: Annotated[int, Field(ge=1, le=2**31)]
    sha256: SHA256


class FetchEvidence(ExecutionModel):
    schema_version: Literal[1] = 1
    mode: Literal["remote_fetch"]
    run_id: str
    node: str
    remote_directory: str
    destination: str
    files: dict[str, FetchedFile]
    total_bytes: Annotated[int, Field(ge=1, le=2**32)]
    remote_hashes_verified: Literal[True]
    local_hashes_verified: Literal[True]
    potcar_fetched: Literal[False]
    wavecar_fetched: Literal[False]

    _safe_run = field_validator("run_id")(safe_id)

    @model_validator(mode="after")
    def complete_inventory(self) -> "FetchEvidence":
        if not REQUIRED_FETCH_FILES.issubset(self.files):
            missing = sorted(REQUIRED_FETCH_FILES - set(self.files))
            raise ValueError(f"fetch evidence is missing required files: {missing}")
        if any(name not in FETCH_ALLOWED_FILES for name in self.files):
            raise ValueError("fetch evidence contains a non-allowlisted filename")
        if sum(item.size for item in self.files.values()) != self.total_bytes:
            raise ValueError("fetch evidence total_bytes does not match its inventory")
        return self


class ArtifactReleaseEvidence(ExecutionModel):
    schema_version: Literal[1] = 1
    mode: Literal["artifact_release"] = "artifact_release"
    artifact: Literal["CHGCAR"] = "CHGCAR"
    workflow_id: str
    source_step: str
    target_step: str
    source_run_id: str
    target_run_id: str
    node: str
    source_remote_directory: str
    target_remote_directory: str
    artifact_size: Annotated[int, Field(ge=1, le=2**31)]
    artifact_sha256: SHA256
    workflow_manifest_sha256: SHA256
    source_step_manifest_sha256: SHA256
    target_step_manifest_sha256: SHA256
    source_run_record_sha256: SHA256
    source_status_sha256: SHA256
    fetch_report_sha256: SHA256
    target_transfer_report_sha256: SHA256
    scientific_result_sha256: SHA256
    source_completed: Literal[True] = True
    source_converged: Literal[True] = True
    dependency_compatible: Literal[True] = True
    remote_source_hash_verified: Literal[True] = True
    remote_target_hash_verified: Literal[True] = True
    promoted_at: datetime
    report_sha256: str = Field(default="", pattern=r"^([0-9a-f]{64})?$")

    _safe_ids = field_validator(
        "workflow_id", "source_step", "target_step", "source_run_id", "target_run_id"
    )(safe_id)

    @field_validator("promoted_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("artifact release timestamp requires a timezone")
        return value

    @model_validator(mode="after")
    def distinct_steps_and_runs(self) -> "ArtifactReleaseEvidence":
        if self.source_step == self.target_step or self.source_run_id == self.target_run_id:
            raise ValueError("artifact release requires distinct source/target steps and runs")
        return self


def _regular_json(path: Path, *, label: str) -> tuple[bytes, dict]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactReleaseError(f"{label} must be a regular local JSON file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactReleaseError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactReleaseError(f"{label} must contain a JSON object")
    return payload, value


def _regular_digest(path: Path, *, expected: FetchedFile, root: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise ArtifactReleaseError("FETCH_LOCAL_VERIFY: fetched evidence is not a regular file")
    size, sha256 = _regular_file_digest(path)
    if size != expected.size or sha256 != expected.sha256:
        raise ArtifactReleaseError("FETCH_LOCAL_VERIFY: fetched file hash/size changed")


def _stamp(evidence: ArtifactReleaseEvidence) -> ArtifactReleaseEvidence:
    payload = evidence.model_dump(mode="json")
    payload.pop("report_sha256", None)
    return evidence.model_copy(update={"report_sha256": digest(json_bytes(payload))})


def artifact_release_valid(evidence: ArtifactReleaseEvidence) -> bool:
    payload = evidence.model_dump(mode="json")
    claimed = payload.pop("report_sha256", "")
    return bool(claimed) and hmac.compare_digest(claimed, digest(json_bytes(payload)))


def _validate_status(
    status_payload: bytes,
    status: dict,
    *,
    source: RunRecord,
    source_workdir: str,
) -> str:
    required = {
        "schema_version": 1,
        "mode": "remote_status",
        "run_id": source.run_id,
        "node": source.node,
        "remote_directory": source_workdir,
        "remote_pid": source.remote_pid,
        "state": "completed",
        "pid_alive": False,
        "pid_cwd_observation": "unknown",
        "scientific_converged": None,
    }
    if any(status.get(key) != value for key, value in required.items()):
        raise ArtifactReleaseError("SOURCE_STATUS_MISMATCH: source run is not cleanly completed")
    if status.get("blockers") != []:
        raise ArtifactReleaseError("SOURCE_STATUS_BLOCKED: source status has blockers")
    return digest(status_payload)


def _validate_fetch(
    report_payload: bytes,
    report: FetchEvidence,
    *,
    source: RunRecord,
    source_workdir: str,
) -> tuple[Path, str]:
    if (
        report.run_id != source.run_id
        or report.node != source.node
        or report.remote_directory != source_workdir
    ):
        raise ArtifactReleaseError("FETCH_EVIDENCE_MISMATCH: fetch is from another run")
    directory = Path(report.destination)
    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactReleaseError("FETCH_LOCAL_VERIFY: fetched directory is unavailable")
    directory = directory.resolve()
    for name, item in report.files.items():
        _regular_digest(directory / name, expected=item, root=directory)
    return directory, digest(report_payload)


def _release_body(
    *,
    source_workdir: str,
    target_workdir: str,
    source_pid: int,
    fetched: FetchEvidence,
    target_hashes: dict[str, str],
) -> str:
    source = shlex.quote(source_workdir)
    target = shlex.quote(target_workdir)
    expected_charge = fetched.files["CHGCAR"]
    temp_name = f".magvasp-CHGCAR-{expected_charge.sha256}.tmp"
    temp = shlex.quote(f"{target_workdir}/{temp_name}")
    destination = shlex.quote(f"{target_workdir}/CHGCAR")
    lines = [
        _directory_guard(source_workdir),
        _directory_guard(target_workdir),
        f"PID={source_pid}",
        "if kill -0 \"$PID\" 2>/dev/null; then",
        "  PID_CWD=\"$(readlink -f -- /proc/$PID/cwd 2>/dev/null || true)\"",
        f"  test \"$PID_CWD\" != {source} || exit 101",
        "fi",
    ]
    # Rehash every fetched source used to establish convergence/provenance.
    for name in sorted(fetched.files):
        path = shlex.quote(f"{source_workdir}/{name}")
        expected = fetched.files[name]
        lines.extend(
            [
                f"test -f {path} && test ! -L {path} || exit 102",
                f"test \"$(stat -c %s -- {path})\" = {expected.size} || exit 103",
                f"test \"$(sha256sum -- {path} | cut -d' ' -f1)\" = {expected.sha256} || exit 104",
            ]
        )
    # Ensure the dependent inputs are exactly the transferred, reviewed bytes.
    for name, expected in sorted(target_hashes.items()):
        path = shlex.quote(f"{target_workdir}/{name}")
        lines.extend(
            [
                f"test -f {path} && test ! -L {path} || exit 105",
                f"test \"$(sha256sum -- {path} | cut -d' ' -f1)\" = {expected} || exit 106",
            ]
        )
    lines.extend(
        [
            f"test ! -e {destination} && test ! -L {destination} || exit 107",
            f"test ! -e {temp} && test ! -L {temp} || exit 108",
            f"cp -- {shlex.quote(source_workdir + '/CHGCAR')} {temp}",
            f"test \"$(stat -c %s -- {temp})\" = {expected_charge.size} || exit 109",
            f"test \"$(sha256sum -- {temp} | cut -d' ' -f1)\" = {expected_charge.sha256} || exit 110",
            # link is the no-overwrite publication boundary; then remove the temp name.
            f"ln -- {temp} {destination}",
            f"rm -- {temp}",
            f"test \"$(sha256sum -- {destination} | cut -d' ' -f1)\" = {expected_charge.sha256} || exit 111",
        ]
    )
    # Detect any source mutation while promotion was in progress.
    for name in sorted(fetched.files):
        path = shlex.quote(f"{source_workdir}/{name}")
        lines.append(
            f"test \"$(sha256sum -- {path} | cut -d' ' -f1)\" = {fetched.files[name].sha256} || exit 112"
        )
    lines.extend(
        [
            "printf 'MAGVASP_ARTIFACT_RELEASE_V1\\n'",
            f"printf 'artifact=CHGCAR|{expected_charge.size}|{expected_charge.sha256}\\n'",
            "printf 'MAGVASP_ARTIFACT_RELEASE_END\\n'",
        ]
    )
    return "\n".join(lines)


def _parse_release(output: bytes, *, expected: FetchedFile) -> None:
    try:
        lines = output.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ArtifactReleaseError("ARTIFACT_RELEASE_PROTOCOL: output is not ASCII") from exc
    wanted = [
        "MAGVASP_ARTIFACT_RELEASE_V1",
        f"artifact=CHGCAR|{expected.size}|{expected.sha256}",
        "MAGVASP_ARTIFACT_RELEASE_END",
    ]
    if lines != wanted:
        raise ArtifactReleaseError("ARTIFACT_RELEASE_PROTOCOL: unexpected output")


def release_chgcar(
    *,
    bundle: Path,
    profile: ClusterProfile,
    source_run: RunRecord,
    source_status: Path,
    fetch_report: Path,
    target_transfer_report: Path,
    target_step: str,
    confirm: str,
    runner=None,
    now: datetime | None = None,
) -> ArtifactReleaseEvidence:
    """Validate a completed SCF and copy its CHGCAR to one declared dependent."""

    if confirm != ARTIFACT_RELEASE_CONFIRM:
        raise ArtifactReleaseError(
            "artifact release requires the exact phrase I_CONFIRM_ARTIFACT_RELEASE"
        )
    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    if not profile.execution_enabled:
        raise ArtifactReleaseError("REMOTE_EXECUTION_DISABLED")
    source_run = RunRecord.model_validate_json(source_run.model_dump_json())
    if not _record_hash_valid(source_run):
        raise ArtifactReleaseError("RUN_RECORD_INTEGRITY: source record hash mismatch")
    profile, source_run, source_workdir = _validated_context(profile, source_run)
    if source_run.node is None or source_run.remote_pid is None:
        raise ArtifactReleaseError("SOURCE_RUN_INCOMPLETE")

    bundle = bundle.resolve(strict=True)
    workflow, target, _ = _read_step(bundle, safe_id(target_step))
    workflow_id = workflow["workflow_id"]
    dependencies = target["status"]["dependencies"]
    if len(dependencies) != 1 or dependencies[0].get("artifact") != "CHGCAR":
        raise ArtifactReleaseError("DEPENDENCY_UNSUPPORTED: target needs one CHGCAR")
    source_step = safe_id(dependencies[0]["from_step"])
    _, source_manifest, _ = _read_step(bundle, source_step)
    source_parts = PurePosixPath(source_workdir).parts
    if (
        source_parts[-1] != source_step
        or source_parts[-2] != f"{workflow_id}--{source_run.run_id}"
    ):
        raise ArtifactReleaseError("SOURCE_RUN_BINDING: run does not match workflow/source step")

    status_payload, status = _regular_json(source_status, label="source status")
    status_sha256 = _validate_status(
        status_payload, status, source=source_run, source_workdir=source_workdir
    )
    fetch_payload, fetch_value = _regular_json(fetch_report, label="fetch report")
    try:
        fetched = FetchEvidence.model_validate(fetch_value)
    except ValueError as exc:
        raise ArtifactReleaseError("FETCH_EVIDENCE_INVALID") from exc
    fetched_dir, fetch_sha256 = _validate_fetch(
        fetch_payload, fetched, source=source_run, source_workdir=source_workdir
    )
    if digest((fetched_dir / "manifest.json").read_bytes()) != workflow["steps"][source_step]["manifest_sha256"]:
        raise ArtifactReleaseError("SOURCE_MANIFEST_MISMATCH")
    if digest((fetched_dir / "POSCAR").read_bytes()) != source_manifest["files"]["POSCAR"]:
        raise ArtifactReleaseError("SOURCE_POSCAR_MISMATCH")
    result = analyze_calculation(fetched_dir)
    if not result.converged:
        raise ArtifactReleaseError("SOURCE_NOT_CONVERGED")
    parsed_sources = {
        "OUTCAR": result.outcar.source_sha256 if result.outcar else None,
        "OSZICAR": result.oszicar.source_sha256 if result.oszicar else None,
        "vasprun.xml": result.vasprun.source_sha256 if result.vasprun else None,
    }
    if any(
        parsed_sources[name] != fetched.files[name].sha256
        for name in parsed_sources
    ):
        raise ArtifactReleaseError("SOURCE_RESULT_HASH_MISMATCH")
    if any(not check.consistent for check in result.cross_checks):
        raise ArtifactReleaseError("SOURCE_RESULT_CROSSCHECK_FAILED")

    transfer_payload, transfer_value = _regular_json(
        target_transfer_report, label="target transfer report"
    )
    try:
        transfer = transfer_report_model(transfer_value)
    except (ValueError, KeyError, TypeError) as exc:
        raise ArtifactReleaseError("TARGET_TRANSFER_INVALID") from exc
    target_run_id = safe_id(transfer.run_id)
    target_root = _safe_remote_path(transfer.remote_directory, label="target_remote_directory")
    expected_root = f"{profile.remote_root}/{workflow_id}--{target_run_id}"
    if (
        transfer.mode != "transfer"
        or transfer.workflow_id != workflow_id
        or transfer.node != source_run.node
        or Path(transfer.local_bundle).resolve() != bundle
        or transfer.remote_directory != expected_root
        or not transfer.transferred
        or not transfer.remote_hashes_verified
        or not transfer.remote_directory_created
        or not transfer.execution_enabled
        or transfer.ready_to_transfer
        or transfer.blockers
        or source_run.run_id == target_run_id
    ):
        raise ArtifactReleaseError("TARGET_TRANSFER_MISMATCH")
    transferred = {item.relative_path: item.sha256 for item in transfer.files}
    if len(transferred) != len(transfer.files):
        raise ArtifactReleaseError("TARGET_TRANSFER_DUPLICATE_PATH")
    if transferred != bundle_transfer_hashes(bundle):
        raise ArtifactReleaseError("TARGET_TRANSFER_INVENTORY_MISMATCH")
    target_hashes = {
        name: target["files"][name]
        for name in ("POSCAR", "POTCAR", "INCAR", "KPOINTS")
    }
    target_hashes["manifest.json"] = workflow["steps"][target_step]["manifest_sha256"]
    if any(
        transferred.get(f"{target_step}/{name}") != value
        for name, value in target_hashes.items()
    ):
        raise ArtifactReleaseError("TARGET_TRANSFER_HASH_MISMATCH")

    target_workdir = _safe_remote_path(
        f"{target_root}/{target_step}", label="target_remote_directory"
    )
    body = _release_body(
        source_workdir=source_workdir,
        target_workdir=target_workdir,
        source_pid=source_run.remote_pid,
        fetched=fetched,
        target_hashes=target_hashes,
    )
    stdout = _execute(
        runner,
        _ssh_argv(profile, source_run.node),
        _fixed_script(profile, source_run.node, body),
        timeout=RELEASE_TIMEOUT_SECONDS,
        output_limit=MAX_COMMAND_OUTPUT_BYTES,
        label="ARTIFACT_RELEASE",
    )
    _parse_release(stdout, expected=fetched.files["CHGCAR"])
    promoted_at = now or datetime.now(UTC)
    evidence = ArtifactReleaseEvidence(
        workflow_id=workflow_id,
        source_step=source_step,
        target_step=target_step,
        source_run_id=source_run.run_id,
        target_run_id=target_run_id,
        node=source_run.node,
        source_remote_directory=source_workdir,
        target_remote_directory=target_workdir,
        artifact_size=fetched.files["CHGCAR"].size,
        artifact_sha256=fetched.files["CHGCAR"].sha256,
        workflow_manifest_sha256=digest((bundle / "workflow.manifest.json").read_bytes()),
        source_step_manifest_sha256=workflow["steps"][source_step]["manifest_sha256"],
        target_step_manifest_sha256=workflow["steps"][target_step]["manifest_sha256"],
        source_run_record_sha256=source_run.record_sha256,
        source_status_sha256=status_sha256,
        fetch_report_sha256=fetch_sha256,
        target_transfer_report_sha256=digest(transfer_payload),
        scientific_result_sha256=result.result_sha256,
        promoted_at=promoted_at,
    )
    return _stamp(evidence)


def release_matches_target(
    evidence: ArtifactReleaseEvidence,
    *,
    bundle: Path,
    workflow_id: str,
    target_step: str,
    target_run_id: str,
    node: str,
    target_remote_directory: str,
    target_transfer_report_sha256: str,
) -> bool:
    """Check immutable release evidence immediately before dependent launch."""

    if not artifact_release_valid(evidence):
        return False
    manifest = json.loads((bundle / "workflow.manifest.json").read_text(encoding="utf-8"))
    try:
        target = json.loads((bundle / target_step / "manifest.json").read_text(encoding="utf-8"))
        dependencies = target["status"]["dependencies"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return (
        len(dependencies) == 1
        and dependencies[0].get("artifact") == "CHGCAR"
        and evidence.workflow_id == workflow_id == manifest.get("workflow_id")
        and evidence.source_step == dependencies[0].get("from_step")
        and evidence.target_step == target_step
        and evidence.target_run_id == target_run_id
        and evidence.node == node
        and evidence.target_remote_directory == target_remote_directory
        and evidence.target_transfer_report_sha256 == target_transfer_report_sha256
        and evidence.workflow_manifest_sha256
        == digest((bundle / "workflow.manifest.json").read_bytes())
        and evidence.source_step_manifest_sha256
        == manifest["steps"][evidence.source_step]["manifest_sha256"]
        and evidence.target_step_manifest_sha256
        == manifest["steps"][target_step]["manifest_sha256"]
    )
