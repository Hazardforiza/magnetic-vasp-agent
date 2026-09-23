"""Build a typed workflow request from mCIF files. MAGMOM is never caller-supplied."""

from __future__ import annotations

import json
import os
from pathlib import Path

from magnetic_vasp_agent.input_review import matching_lessons
from magnetic_vasp_agent.kpoints_models import KpointsRecipeRequest
from magnetic_vasp_agent.mcif import map_moments, read_mcif
from magnetic_vasp_agent.mcif_models import McifIncarRequest
from magnetic_vasp_agent.models import CalculationGoal, CalculationTask, MagneticMode
from magnetic_vasp_agent.workflow_models import ArtifactDependency, CalculationStep, Workflow, safe_id


class McifWorkflowError(ValueError):
    """Raised when an mCIF-to-workflow plan is refused."""


def resolve_magnetic_mode(collinear: bool, requested: str) -> MagneticMode:
    if requested == "auto":
        return MagneticMode.COLLINEAR if collinear else MagneticMode.NONCOLLINEAR
    mode = MagneticMode(requested)
    if mode == MagneticMode.COLLINEAR and not collinear:
        raise McifWorkflowError(
            "MCIF_NONCOLLINEAR: this candidate has transverse moments; "
            "collinear mode is refused"
        )
    return mode


def _step_id(stem: str, *, prefix: str, unique: bool) -> str:
    raw = f"{prefix}-{stem}" if unique else prefix
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)
    return safe_id(cleaned)


def _posix_relpath(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target.resolve(), start=start.resolve())).as_posix()


def plan_workflow_from_mcif(
    mcif_paths: list[Path],
    output: Path,
    *,
    workflow_id: str | None = None,
    task: str = "static",
    magnetic_mode: str = "auto",
    encut: float = 600.0,
    spacing: float = 0.25,
    include_dos: bool = False,
    dos_spacing: float = 0.15,
    notes: str | None = None,
    review_log: Path | None = None,
) -> dict:
    if not mcif_paths:
        raise McifWorkflowError("at least one mCIF path is required")
    if include_dos and task != "static":
        raise McifWorkflowError("DOS follow-up is only attached to a static SCF")
    if output.exists() or output.is_symlink():
        raise FileExistsError("workflow request exists; choose a new path")
    if magnetic_mode not in {"auto", "collinear", "noncollinear", "soc"}:
        raise McifWorkflowError("magnetic_mode must be auto, collinear, noncollinear, or soc")

    paths = [path.resolve() for path in mcif_paths]
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        raise McifWorkflowError("mCIF filenames must have unique stems")
    many = len(paths) > 1
    goal_task = CalculationTask(task)
    if goal_task == CalculationTask.BAND:
        raise McifWorkflowError(
            "band steps need an explicit high-symmetry path; they are not auto-generated"
        )
    if goal_task == CalculationTask.DOS:
        raise McifWorkflowError("use task=static with include_dos, not task=dos")

    resolved_id = safe_id(workflow_id or (stems[0] if not many else "mcif-bundle"))
    request_parent = output.resolve().parent
    request_parent.mkdir(parents=True, exist_ok=True)

    steps: list[CalculationStep] = []
    preview: list[dict] = []
    for path, stem in zip(paths, stems, strict=True):
        candidate = read_mcif(path)
        mode = resolve_magnetic_mode(candidate.collinear, magnetic_mode)
        moment_mode = (
            "collinear_projection"
            if mode == MagneticMode.COLLINEAR
            else "preserve_cartesian"
        )
        goal = CalculationGoal(task=goal_task, magnetic_mode=mode, notes=notes)
        incar = McifIncarRequest(
            goal=goal,
            encut=encut,
            moment_source="mcif",
            moment_mode=moment_mode,
            saxis=[0.0, 0.0, 1.0],
        )
        resolved, mapping = map_moments(candidate, incar)
        posix = _posix_relpath(path, request_parent)
        scf_id = _step_id(stem, prefix="scf", unique=many)
        kpoints = KpointsRecipeRequest.model_validate(
            {"goal": goal.model_dump(mode="json"), "spacing": spacing, "centering": "gamma"}
        )
        steps.append(
            CalculationStep(
                id=scf_id,
                candidate_id=candidate.candidate_id,
                mcif=posix,
                incar=incar,
                kpoints=kpoints,
            )
        )
        if include_dos:
            dos_goal = goal.model_copy(update={"task": CalculationTask.DOS})
            dos_incar = incar.model_copy(update={"goal": dos_goal})
            dos_id = _step_id(stem, prefix="dos", unique=many)
            steps.append(
                CalculationStep(
                    id=dos_id,
                    candidate_id=candidate.candidate_id,
                    mcif=posix,
                    incar=dos_incar,
                    kpoints=KpointsRecipeRequest.model_validate(
                        {
                            "goal": dos_goal.model_dump(mode="json"),
                            "spacing": dos_spacing,
                            "centering": "gamma",
                        }
                    ),
                    dependencies=[ArtifactDependency(artifact="CHGCAR", from_step=scf_id)],
                )
            )
        preview.append(
            {
                "candidate_id": candidate.candidate_id,
                "mcif": posix,
                "species_order": candidate.structure.species_order,
                "counts": candidate.structure.counts,
                "declared_bns": candidate.declared_bns,
                "collinear": candidate.collinear,
                "magnetic_mode": mode.value,
                "moment_mode": moment_mode,
                "magmom": mapping.magmom,
                "executable": "vasp_std" if mode == MagneticMode.COLLINEAR else "vasp_ncl",
                "encut": encut,
                "ediff": resolved.ediff,
            }
        )

    workflow = Workflow(workflow_id=resolved_id, steps=steps)
    payload = workflow.model_dump(mode="json", exclude_unset=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    first = preview[0] if preview else {}
    prior = [
        item.model_dump(mode="json")
        for item in matching_lessons(
            species=first.get("species_order"),
            task=goal_task.value,
            magnetic_mode=first.get("magnetic_mode"),
            log_path=review_log,
        )
    ]
    return {
        "workflow_id": resolved_id,
        "request_path": str(output.resolve()),
        "step_ids": [step.id for step in steps],
        "scientific_converged": False,
        "awaiting_human": True,
        "preview": preview,
        "prior_lessons": prior,
        "notes": (
            "MAGMOM was mapped from the mCIF, not invented. "
            "This is an input package only; it is not a converged result. "
            "A human must review the built bundle before it is treated as accepted."
        ),
    }
