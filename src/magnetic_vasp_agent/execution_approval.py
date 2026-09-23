"""Issue a time-bounded human resource grant from locally verified evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    ExecutionRequest,
    ProbeSnapshot,
    ResourceGrant,
)
from magnetic_vasp_agent.execution_probe import assess_resources
from magnetic_vasp_agent.input_review import CARD_NAME
from magnetic_vasp_agent.input_review_models import (
    REVIEWER_BLOCKLIST,
    InputReviewCard,
)
from magnetic_vasp_agent.workflow import INPUT_NAMES, digest, inspect_bundle, json_bytes

RESOURCE_GRANT_CONFIRM = "I_APPROVE_RESOURCE_GRANT"
DEFAULT_GRANT_MINUTES = 30
MAX_GRANT_MINUTES = 120
_RESOLVED_BY_HUMAN_GRANT = frozenset({"ALLOCATION_UNCONFIRMED"})


def _human_name(value: str, *, field: str) -> str:
    name = value.strip()
    if not name or name.casefold() in REVIEWER_BLOCKLIST:
        raise ValueError(f"{field} must be a named human, not the agent")
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise ValueError(f"{field} must not contain control characters")
    return name


def validate_human_approver(value: str) -> str:
    """Normalize one named human approver for approval-bound adapters."""

    return _human_name(value, field="approver")


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path


def _load_review(bundle: Path, workflow: dict) -> InputReviewCard:
    path = _regular_file(bundle / CARD_NAME, label="input review")
    card = InputReviewCard.model_validate_json(path.read_bytes())
    if (
        card.status != "reviewed"
        or card.decision != "accept"
        or card.awaiting_human
        or not card.integrity_valid
    ):
        raise ValueError("INPUT_REVIEW_REQUIRED: input.review.json must be reviewed and accepted")
    _human_name(card.reviewer or "", field="input reviewer")
    if card.workflow_id != workflow.get("workflow_id"):
        raise ValueError("INPUT_REVIEW_MISMATCH: review workflow does not match bundle")
    try:
        reviewed_bundle = Path(card.bundle).resolve(strict=True)
    except OSError as exc:
        raise ValueError("INPUT_REVIEW_MISMATCH: reviewed bundle path is unavailable") from exc
    if reviewed_bundle != bundle:
        raise ValueError("INPUT_REVIEW_MISMATCH: review belongs to a different bundle")

    reviewed_steps = {step.step_id: step for step in card.steps}
    if len(reviewed_steps) != len(card.steps) or set(reviewed_steps) != set(workflow["steps"]):
        raise ValueError("INPUT_REVIEW_MISMATCH: review does not cover every workflow step")
    for step_id, record in workflow["steps"].items():
        if reviewed_steps[step_id].file_hashes != record["files"]:
            raise ValueError(
                f"INPUT_REVIEW_MISMATCH: reviewed hashes changed for step {step_id}"
            )
    return card


def _grant_id(request: ExecutionRequest, approved_at: datetime) -> str:
    request_hash = digest(json_bytes(request.model_dump(mode="json")))[:12]
    stamp = approved_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"grant-{request_hash}-{stamp}"


def build_resource_grant(
    bundle: Path,
    profile: ClusterProfile,
    request: ExecutionRequest,
    snapshot: ProbeSnapshot,
    *,
    approver: str,
    confirm: str,
    now: datetime | None = None,
    valid_for: timedelta = timedelta(minutes=DEFAULT_GRANT_MINUTES),
    grant_id: str | None = None,
) -> ResourceGrant:
    """Revalidate local evidence and build one complete human resource grant."""

    if confirm != RESOURCE_GRANT_CONFIRM:
        raise ValueError(
            "resource grant requires the exact phrase I_APPROVE_RESOURCE_GRANT"
        )
    approver = validate_human_approver(approver)
    approved_at = now or datetime.now(UTC)
    if approved_at.utcoffset() is None:
        raise ValueError("resource grant time requires a timezone")
    if valid_for <= timedelta(0) or valid_for > timedelta(minutes=MAX_GRANT_MINUTES):
        raise ValueError("resource grant validity must be between 1 second and 120 minutes")

    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    request = ExecutionRequest.model_validate_json(request.model_dump_json())
    snapshot = ProbeSnapshot.model_validate_json(snapshot.model_dump_json())
    root = bundle.resolve(strict=True)
    if bundle.is_symlink() or not root.is_dir():
        raise ValueError("workflow bundle must be a regular directory, not a symlink")

    inspection = inspect_bundle(root)
    workflow_path = _regular_file(
        root / "workflow.manifest.json", label="workflow manifest"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    if inspection["workflow_id"] != workflow.get("workflow_id"):
        raise ValueError("WORKFLOW_MISMATCH: inspection and manifest disagree")
    if request.step_id not in workflow["steps"]:
        raise ValueError("UNKNOWN_STEP: request step is not in the workflow")
    _load_review(root, workflow)

    step_dir = root / request.step_id
    step_path = _regular_file(step_dir / "manifest.json", label="step manifest")
    step = json.loads(step_path.read_text(encoding="utf-8"))
    input_hashes = {
        name: digest(_regular_file(step_dir / name, label=name).read_bytes())
        for name in INPUT_NAMES
    }
    if input_hashes != step.get("files"):
        raise ValueError("INPUT_HASH_MISMATCH: current files differ from step manifest")

    assessment = assess_resources(snapshot, request, now=approved_at, profile=profile)
    unresolved = [
        blocker
        for blocker in assessment.blockers
        if blocker.code not in _RESOLVED_BY_HUMAN_GRANT
    ]
    if unresolved:
        codes = ", ".join(blocker.code for blocker in unresolved)
        raise ValueError(f"RESOURCE_ASSESSMENT_BLOCKED: {codes}")
    cpu_ids = (
        list(request.cpu_ids)
        if request.cpu_ids is not None
        else list(assessment.suggested_cpu_ids)
    )
    if len(cpu_ids) != request.mpi_ranks:
        raise ValueError("CPU_BINDING_REQUIRED: grant needs one CPU per MPI rank")

    request_sha256 = digest(json_bytes(request.model_dump(mode="json")))
    return ResourceGrant(
        grant_id=grant_id or _grant_id(request, approved_at),
        workflow_id=workflow["workflow_id"],
        step_id=request.step_id,
        run_id=request.run_id,
        workflow_manifest_sha256=digest(workflow_path.read_bytes()),
        step_manifest_sha256=digest(step_path.read_bytes()),
        run_sha256=request_sha256,
        input_sha256=input_hashes,
        node=request.node,
        mpi_ranks=request.mpi_ranks,
        cpu_ids=cpu_ids,
        valid_from=approved_at,
        valid_until=approved_at + valid_for,
        approver=approver,
        approved_at=approved_at,
    )


def issue_resource_grant(
    bundle: Path,
    profile: ClusterProfile,
    request: ExecutionRequest,
    snapshot: ProbeSnapshot,
    output: Path,
    *,
    approver: str,
    confirm: str,
    now: datetime | None = None,
    valid_for: timedelta = timedelta(minutes=DEFAULT_GRANT_MINUTES),
    grant_id: str | None = None,
) -> ResourceGrant:
    """Build and atomically create a grant file; existing paths are refused."""

    if output.exists() or output.is_symlink():
        raise FileExistsError("resource grant output exists; refusing to overwrite")
    grant = build_resource_grant(
        bundle,
        profile,
        request,
        snapshot,
        approver=approver,
        confirm=confirm,
        now=now,
        valid_for=valid_for,
        grant_id=grant_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(json_bytes(grant.model_dump(mode="json")))
    return grant
