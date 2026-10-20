"""Build review cards and record human decisions. The agent cannot self-approve."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from magnetic_vasp_agent.execution_site import reject_secret_text
from magnetic_vasp_agent.input_review_models import (
    InputReviewCard,
    InputReviewRecord,
    InputReviewStep,
    PriorLesson,
)
from magnetic_vasp_agent.workflow import inspect_bundle

DEFAULT_REVIEW_LOG = Path("var/reviews/input-reviews.jsonl")
DEFAULT_USAGE_DIR = Path("var/reviews/usage")
CARD_NAME = "input.review.json"
_REVIEWER_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _step_card(bundle: Path, step_id: str) -> InputReviewStep:
    manifest = json.loads((bundle / step_id / "manifest.json").read_text(encoding="utf-8"))
    goal = manifest.get("goal") or {}
    incar = manifest.get("incar") or {}
    potcar = manifest.get("potcar") or {}
    kpoints = (manifest.get("request") or {}).get("kpoints") or {}
    sampling = kpoints.get("sampling") or {}
    if sampling.get("mesh"):
        summary = f"{sampling.get('mode', 'mesh')} {'x'.join(str(n) for n in sampling['mesh'])}"
    elif kpoints.get("spacing") is not None:
        summary = f"spacing {kpoints['spacing']} {kpoints.get('centering') or ''}".strip()
    else:
        summary = sampling.get("mode")
    structure = (manifest.get("candidate") or {}).get("structure") or {}
    species = list(structure.get("species_order") or potcar.get("species_order") or [])
    counts = list(structure.get("counts") or [])
    if not species:
        mapping = list(manifest.get("atom_mapping") or [])
        species = list(dict.fromkeys(atom.get("species") for atom in mapping if atom.get("species")))
        counts = [sum(1 for atom in mapping if atom.get("species") == name) for name in species]
    return InputReviewStep(
        step_id=step_id,
        task=str(goal.get("task") or "unknown"),
        magnetic_mode=str(goal.get("magnetic_mode") or "unknown"),
        executable=str((manifest.get("status") or {}).get("executable") or "unknown"),
        species_order=species,
        counts=counts,
        encut=incar.get("ENCUT"),
        magmom=list(incar.get("MAGMOM") or []),
        potcar_names=list(potcar.get("potential_names") or []),
        kpoints_summary=summary,
        file_hashes=dict(manifest.get("files") or {}),
    )


def matching_lessons(
    *,
    species: list[str] | None = None,
    task: str | None = None,
    magnetic_mode: str | None = None,
    reviewer: str | None = None,
    log_path: Path | None = None,
    limit: int = 5,
) -> list[PriorLesson]:
    lessons: list[PriorLesson] = []
    for record in load_review_history(
        log_path,
        species=species,
        task=task,
        magnetic_mode=magnetic_mode,
        reviewer=reviewer,
        limit=limit,
    ):
        text = (record.lesson or record.notes or "").strip()
        if not text:
            continue
        lessons.append(
            PriorLesson(
                workflow_id=record.workflow_id,
                decision=record.decision or "reviewed",
                reviewer=record.reviewer,
                lesson=text,
                tags=list(record.tags),
            )
        )
    return lessons


def _priors_for_steps(steps: list[InputReviewStep], log_path: Path | None) -> list[PriorLesson]:
    first = steps[0] if steps else None
    if first is None:
        return []
    return matching_lessons(
        species=first.species_order or None,
        task=first.task,
        magnetic_mode=first.magnetic_mode,
        log_path=log_path,
    )


def build_review_card(bundle: Path, *, log_path: Path | None = None) -> InputReviewCard:
    root = bundle.resolve()
    inspection = inspect_bundle(root)
    existing = root / CARD_NAME
    if existing.is_file() and not existing.is_symlink():
        card = InputReviewCard.model_validate_json(existing.read_bytes())
        if card.status == "pending":
            card = card.model_copy(update={"prior_lessons": _priors_for_steps(card.steps, log_path)})
        return card
    workflow = json.loads((root / "workflow.manifest.json").read_text(encoding="utf-8"))
    order = list(workflow.get("order") or inspection.get("steps") or [])
    if not order and isinstance(workflow.get("steps"), dict):
        order = list(workflow["steps"])
    steps = [_step_card(root, step_id) for step_id in order]
    card = InputReviewCard(
        workflow_id=str(inspection["workflow_id"]),
        bundle=str(root),
        integrity_valid=bool(inspection["integrity_valid"]),
        awaiting_human=True,
        steps=steps,
        prior_lessons=_priors_for_steps(steps, log_path),
    )
    existing.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return card


def record_input_review(
    bundle: Path,
    *,
    reviewer: str,
    decision: str,
    notes: str | None = None,
    lesson: str | None = None,
    tags: list[str] | None = None,
    log_path: Path | None = None,
) -> InputReviewCard:
    card = build_review_card(bundle, log_path=log_path)
    if not card.integrity_valid:
        raise ValueError("refusing to review a bundle that failed integrity checks")
    if decision in {"reject", "revise"} and not (notes and notes.strip()):
        raise ValueError("reject and revise reviews require notes the agent can learn from")
    notes = reject_secret_text(notes, label="review notes")
    lesson = reject_secret_text(lesson, label="review lesson")
    updated = InputReviewCard.model_validate(
        {
            **card.model_dump(mode="json"),
            "status": "reviewed",
            "awaiting_human": False,
            "decision": decision,
            "reviewer": reviewer,
            "notes": notes,
            "lesson": lesson,
            "tags": tags or [],
            "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    path = Path(card.bundle) / CARD_NAME
    path.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    record = InputReviewRecord.model_validate(updated.model_dump(mode="json"))
    log = log_path or DEFAULT_REVIEW_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record.model_dump_json() + "\n")
    write_usage_note(record, usage_dir=log.parent / "usage" if log_path else None)
    return updated


def load_review_history(
    log_path: Path | None = None,
    *,
    species: list[str] | None = None,
    task: str | None = None,
    magnetic_mode: str | None = None,
    reviewer: str | None = None,
    limit: int = 10,
) -> list[InputReviewRecord]:
    log = log_path or DEFAULT_REVIEW_LOG
    if not log.is_file():
        return []
    wanted = reviewer.strip().casefold() if reviewer else None
    records: list[InputReviewRecord] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = InputReviewRecord.model_validate_json(line)
        if record.status != "reviewed":
            continue
        steps = record.steps
        if species and not any(step.species_order == species for step in steps):
            continue
        if task and not any(step.task == task for step in steps):
            continue
        if magnetic_mode and not any(step.magnetic_mode == magnetic_mode for step in steps):
            continue
        if wanted and (record.reviewer or "").casefold() != wanted:
            continue
        records.append(record)
    return records[-limit:]


def usage_reviewer_name(reviewer: str) -> str:
    token = reviewer.strip()
    if _REVIEWER_FILE.fullmatch(token) is None:
        raise ValueError("reviewer must be a cluster-username-like token for usage files")
    return token


def write_usage_note(record: InputReviewRecord, *, usage_dir: Path | None = None) -> Path:
    """Write a sanitized per-reviewer markdown note for local knowledge only."""

    if not record.reviewer:
        raise ValueError("usage notes require a named human reviewer")
    name = usage_reviewer_name(record.reviewer)
    directory = usage_dir or DEFAULT_USAGE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    jsonl = directory / f"{name}.jsonl"
    with jsonl.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "recorded_at": record.recorded_at,
                    "reviewer": name,
                    "workflow_id": record.workflow_id,
                    "decision": record.decision,
                    "tags": list(record.tags),
                    "lesson": record.lesson,
                    "notes": record.notes,
                    "species": [step.species_order for step in record.steps],
                    "tasks": [step.task for step in record.steps],
                    "magnetic_modes": [step.magnetic_mode for step in record.steps],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    markdown = directory / f"{name}.md"
    markdown.write_text(_render_usage_markdown(name, jsonl), encoding="utf-8", newline="\n")
    return markdown


def _render_usage_markdown(reviewer: str, jsonl: Path) -> str:
    lines = [
        f"# Usage records for {reviewer}",
        "",
        "Local-only member notes. They are not a scientific result, do not store",
        "passwords or keys, and do not authorize cluster launch.",
        "",
    ]
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        when = item.get("recorded_at") or "unknown-time"
        workflow = item.get("workflow_id") or "unknown-workflow"
        lines.extend(
            [
                f"## {when} {workflow}",
                "",
                f"- decision: {item.get('decision')}",
                f"- tags: {', '.join(item.get('tags') or []) or '(none)'}",
                f"- species: {item.get('species')}",
                f"- tasks: {item.get('tasks')}",
                f"- magnetic modes: {item.get('magnetic_modes')}",
            ]
        )
        if item.get("lesson"):
            lines.append(f"- lesson: {item['lesson']}")
        if item.get("notes"):
            lines.append(f"- notes: {item['notes']}")
        lines.append("")
    return "\n".join(lines)
