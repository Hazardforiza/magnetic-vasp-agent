"""Bind immutable input packages to reviewable plans, never to remote execution."""

from __future__ import annotations

import json
import shlex
from datetime import datetime
from pathlib import Path

from magnetic_vasp_agent.execution_models import (
    MPI_PATH,
    VASP_PATHS,
    Blocker,
    ClusterProfile,
    ExecutionRequest,
    ProbeSnapshot,
)
from magnetic_vasp_agent.execution_probe import assess_resources, probe_preview
from magnetic_vasp_agent.incar import read_incar, validate_incar
from magnetic_vasp_agent.incar_models import IncarSpec
from magnetic_vasp_agent.kpoints import read_kpoints, validate_kpoints
from magnetic_vasp_agent.models import CalculationGoal, PotentialPlan
from magnetic_vasp_agent.poscar import read_poscar
from magnetic_vasp_agent.workflow import INPUT_NAMES, digest, inspect_bundle, json_bytes
from magnetic_vasp_agent.workflow_models import CalculationStep, StepStatus, safe_id


def _read_step(bundle: Path, step_id: str) -> tuple[dict, dict, IncarSpec]:
    if (bundle / "workflow.manifest.json").is_symlink():
        raise ValueError("workflow manifest must not be a symlink")
    inspect_bundle(bundle)
    manifest = json.loads((bundle / "workflow.manifest.json").read_text(encoding="utf-8"))
    safe_id(manifest["workflow_id"])
    if step_id not in manifest["steps"]:
        raise ValueError("UNKNOWN_STEP: step is not present in the workflow")
    folder = bundle / step_id
    step = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    status = StepStatus.model_validate(step["status"])
    request = CalculationStep.model_validate_json(json.dumps(step["request"]))
    goal = CalculationGoal.model_validate(step["goal"])
    plan = PotentialPlan.model_validate(step["potential_plan"])
    if (
        status.id != step_id or request.id != step_id
        or request.candidate_id != step["candidate_id"]
        or plan.candidate_id != step["candidate_id"]
        or request.incar.goal != goal or plan.goal != goal
    ):
        raise ValueError("MANIFEST_CONTRACT: step identity or goal mismatch")
    structure = read_poscar(folder / "POSCAR")
    incar = read_incar(folder / "INCAR", legacy_booleans=False).spec
    if incar.model_dump(exclude_none=True) != step["incar"]:
        raise ValueError("MANIFEST_CONTRACT: INCAR differs from recorded parameters")
    if plan.species_order != structure.species_order:
        raise ValueError("MANIFEST_CONTRACT: POSCAR and potential plan species differ")
    if (
        step["potcar"]["species_order"] != structure.species_order
        or step["potcar"]["output_sha256"] != step["files"]["POTCAR"]
    ):
        raise ValueError("MANIFEST_CONTRACT: POTCAR metadata mismatch")
    # Reuse the compiled POTCAR hash; source-library ENMAX is not re-audited here.
    report = validate_incar(incar, structure, goal=goal, executable=status.executable)
    report.raise_for_errors()
    validate_kpoints(read_kpoints(folder / "KPOINTS").spec, structure, goal, incar).raise_for_errors()
    dependencies = [{"artifact": d.artifact, "from_step": d.from_step} for d in status.dependencies]
    if (
        dependencies != [d.model_dump() for d in request.dependencies]
        or sorted(report.required_artifacts) != sorted(d["artifact"] for d in dependencies)
        or status.input_state != ("waiting_for_upstream" if dependencies else "prepared")
        or any(d.sha256 is not None for d in status.dependencies)
    ):
        raise ValueError("MANIFEST_CONTRACT: upstream state or artifact bindings mismatch")
    if not step["validation"]["valid"] or not manifest["validation"]["valid"]:
        raise ValueError("MANIFEST_CONTRACT: input package validation did not pass")
    return manifest, step, incar


def prepare_execution(
    bundle: Path,
    profile: ClusterProfile,
    request: ExecutionRequest,
    snapshot: ProbeSnapshot | None = None,
    *, now: datetime | None = None,
) -> dict:
    profile = ClusterProfile.model_validate_json(profile.model_dump_json())
    request = ExecutionRequest.model_validate_json(request.model_dump_json())
    bundle = bundle.resolve()
    try:
        manifest, step, incar = _read_step(bundle, request.step_id)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("MANIFEST_CONTRACT: malformed workflow bundle") from exc
    blockers = [
        Blocker(code="REMOTE_EXECUTION_DISABLED", message="Step 8 implements preparation only; no launch/upload/cancel API exists"),
        Blocker(code="MPI_CONFINEMENT_UNVERIFIED", message="Intel MPI single-host/pinning behavior still needs an approved smoke test"),
        Blocker(code="SCIENTIFIC_REVIEW_REQUIRED", message="Potential choice, U/J, magnetic state and convergence settings require review"),
        Blocker(code="REMOTE_DIRECTORY_UNVERIFIED", message="Remote storage, quota, permissions and absence of prior runs are unverified"),
    ]
    assessment = None
    cpus = request.cpu_ids
    if snapshot is None:
        blockers.append(Blocker(code="PREFLIGHT_REQUIRED", message="No current node snapshot was supplied"))
        blockers.append(Blocker(code="ALLOCATION_UNCONFIRMED", message="No group resource allocation is established"))
    else:
        assessment = assess_resources(snapshot, request, now=now)
        cpus = assessment.suggested_cpu_ids
        blockers.extend(assessment.blockers)
        if not snapshot.executables[incar.executable]:
            blockers.append(Blocker(code="VASP_UNAVAILABLE", message="Required executable was not observed at the expected path"))
    if not cpus or len(cpus) != request.mpi_ranks:
        blockers.append(Blocker(code="CPU_BINDING_REQUIRED", message="One reviewed CPU per physical core is required"))
        cpus = None
    if profile.remote_root is None:
        blockers.append(Blocker(code="REMOTE_ROOT_REQUIRED", message="Choose and verify a dedicated group-approved run directory"))
    if step["status"]["dependencies"]:
        blockers.append(Blocker(code="UPSTREAM_NOT_VERIFIED", message="Input plans and same-named local artifacts do not establish upstream convergence or compatibility"))
    if incar.SOCSCALE is not None or incar.LWANNIER90:
        blockers.append(Blocker(code="BUILD_CAPABILITY_UNVERIFIED", message="SOCSCALE/Wannier support cannot be inferred from the module name"))
    if incar.NPAR is not None and incar.NCORE is not None:
        blockers.append(Blocker(code="PARALLEL_TAG_CONFLICT", message="Review simultaneous NPAR and NCORE; the executor does not rewrite INCAR"))
    for tag in ("NCORE", "NPAR"):
        value = getattr(incar, tag)
        if value and request.mpi_ranks % value:
            blockers.append(Blocker(code="PARALLEL_DIVISIBILITY", message=f"{tag} must divide the requested pure-MPI rank count in this backend"))
    remote_directory = (
        f"{profile.remote_root}/{manifest['workflow_id']}/{request.run_id}/{request.step_id}"
        if profile.remote_root else None
    )
    mpi_argv = [
        MPI_PATH, "-bootstrap", "fork", "-hosts", request.node,
        "-n", str(request.mpi_ranks), VASP_PATHS[incar.executable],
    ]
    environment = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "I_MPI_PIN": "1"}
    if cpus:
        environment["I_MPI_PIN_PROCESSOR_LIST"] = ",".join(str(i) for i in cpus)
    preview_script = None
    if cpus and remote_directory:
        # An unconditional guard prevents this review artifact from starting computation.
        preview_script = "\n".join([
            "#!/bin/bash", "# REVIEW ONLY. No resource allocation or runtime verification.",
            "exit 78", "set -eu", f'test "$(hostname -s)" = {request.node}',
            'test "$(id -un)" = zyy', f"module load {profile.vasp_module}",
            f"cd -- {shlex.quote(remote_directory)}",
            *[f"export {key}={shlex.quote(value)}" for key, value in environment.items()],
            "# This is not a validated launch script. Parent MPI environment must be audited.",
            shlex.join(mpi_argv), "",
        ])
    return {
        "schema_version": 1, "mode": "execution_preview", "ready_to_run": False,
        "runtime_verified": False, "execution_enabled": False,
        "workflow_id": manifest["workflow_id"], "bundle": str(bundle),
        "workflow_manifest_sha256": digest((bundle / "workflow.manifest.json").read_bytes()),
        "step_manifest_sha256": manifest["steps"][request.step_id]["manifest_sha256"],
        "input_sha256": {name: step["files"][name] for name in INPUT_NAMES},
        "profile": profile.model_dump(mode="json"), "request": request.model_dump(mode="json"),
        "goal": step["goal"], "executable": incar.executable,
        "dependencies": step["status"]["dependencies"],
        "remote_directory": remote_directory, "mpi_argv_candidate": mpi_argv,
        "environment_candidate": environment, "guarded_script_preview": preview_script,
        "probe_preview": probe_preview(profile, request.node),
        "snapshot_sha256": digest(json_bytes(snapshot.model_dump(mode="json"))) if snapshot else None,
        "resources": assessment.model_dump(mode="json") if assessment else None,
        "blockers": [item.model_dump() for item in blockers],
        "limitations": [
            "Hashes are integrity checks, not signatures; a rewritten manifest is not trusted approval",
            "POTCAR contents are never included in this plan; source-library ENMAX is not re-audited here",
            "No artifact transfer, host reservation, automatic retry or convergence validation occurs",
        ],
    }


def write_report(report: dict, output: Path) -> None:
    """Write a new local report, never replace a previous record."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(json_bytes(report))
