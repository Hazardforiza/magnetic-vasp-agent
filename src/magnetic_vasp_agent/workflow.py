"""Deterministic workflow compilation and transactional local materialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.io.vasp.inputs import Poscar

from magnetic_vasp_agent import __version__
from magnetic_vasp_agent.incar import parse_incar, render_incar, validate_incar
from magnetic_vasp_agent.incar_models import IncarSpec
from magnetic_vasp_agent.incar_recipes import build_incar
from magnetic_vasp_agent.kpoints import parse_kpoints, render_kpoints
from magnetic_vasp_agent.kpoints_models import KpointsSpec
from magnetic_vasp_agent.kpoints_recipes import build_kpoints
from magnetic_vasp_agent.models import (
    CandidateComposition,
    PotentialPolicy,
    PotentialRegistry,
    Structure,
)
from magnetic_vasp_agent.poscar import parse_poscar, render_poscar
from magnetic_vasp_agent.potcar.selector import prepare_potcar, select_by_policy
from magnetic_vasp_agent.workflow_models import (
    CalculationStep,
    PendingArtifact,
    StepStatus,
    Workflow,
    safe_id,
)

INPUT_NAMES = ("POSCAR", "POTCAR", "INCAR", "KPOINTS")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


@dataclass
class PreparedStep:
    request: CalculationStep
    structure: Structure
    incar: IncarSpec
    kpoints: KpointsSpec
    payloads: dict[str, bytes] = field(repr=False)
    manifest: dict[str, Any]


@dataclass
class PreparedWorkflow:
    steps: dict[str, PreparedStep] = field(repr=False)
    manifest: dict[str, Any]


def expand_steps(
    workflow: Workflow,
) -> tuple[list[CalculationStep], list[dict[str, Any]]]:
    steps = list(workflow.steps)
    groups = []
    for scan in workflow.scans:
        variants = scan.axes if scan.kind == "mae" else scan.values
        ids = []
        for label, value in variants.items():
            data = scan.base.model_dump(mode="json")
            data["id"] = f"{scan.base.id}-{label}"
            data["incar_options"] = scan.base.incar_options.model_dump(
                exclude_unset=True
            )
            if scan.kind == "mae":
                data["incar"]["saxis"] = value
            else:
                data["incar_options"]["SOCSCALE"] = value
            step = CalculationStep.model_validate(data)
            steps.append(step)
            ids.append(step.id)
        groups.append(
            {
                "kind": scan.kind,
                "method": "self_consistent_soc",
                "steps": ids,
                "varied_tag": "SAXIS" if scan.kind == "mae" else "SOCSCALE",
                "values": variants,
                "results_available": False,
            }
        )
    ids = [step.id.casefold() for step in steps]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "DUPLICATE_STEP: step IDs must be unique, including on Windows"
        )
    return steps, groups


def _cell(structure: Structure):
    values = [
        structure.scale,
        *(x for row in structure.lattice for x in row),
        *(x for row in structure.positions for x in row),
    ]
    if (
        any(not math.isfinite(x) for x in values)
        or abs(np.linalg.det(structure.lattice)) < 1e-12
    ):
        raise ValueError(
            "INVALID_CELL: finite positions and a nonsingular cell are required"
        )
    cell = Poscar.from_str(render_poscar(structure)).structure
    if not math.isfinite(cell.volume) or cell.volume < 1e-12:
        raise ValueError(
            "INVALID_CELL: physical cell volume must be finite and positive"
        )
    return cell


def _same_structure(left: PreparedStep, right: PreparedStep) -> bool:
    if left.request.candidate_id != right.request.candidate_id:
        return False
    if left.structure.site_species != right.structure.site_species:
        return False
    a, b = _cell(left.structure), _cell(right.structure)
    delta = a.frac_coords - b.frac_coords
    return bool(
        np.allclose(a.lattice.matrix, b.lattice.matrix, atol=1e-10, rtol=0)
        and np.allclose(delta - np.rint(delta), 0, atol=1e-10, rtol=0)
    )


def _sampling(spec: KpointsSpec) -> dict:
    return spec.model_dump(exclude={"comment"})


def _read_structure(payload: bytes) -> Structure:
    text = payload.decode("utf-8-sig")
    lines = text.splitlines()
    if len(lines) < 8 or len(lines[1].split()) != 1:
        raise ValueError("POSCAR_LAYOUT: v1 workflows require one scalar scale factor")
    cursor = 8 if lines[7].strip()[:1].lower() == "s" else 7
    if len(lines) <= cursor or lines[cursor].strip()[:1].lower() not in {"c", "k", "d"}:
        raise ValueError(
            "POSCAR_LAYOUT: expected an explicit Direct or Cartesian coordinate mode"
        )
    structure = parse_poscar(text)
    tail = [line for line in lines[cursor + 1 + structure.num_atoms :] if line.strip()]
    # Restrict the workflow boundary to the already modeled POSCAR subset.
    if tail and (len(tail) != structure.num_atoms or structure.velocities is None):
        raise ValueError("POSCAR_LAYOUT: unsupported trailing POSCAR data")
    return structure


def _compile_step(
    step: CalculationStep,
    workflow: Workflow,
    registry: PotentialRegistry,
    policy: PotentialPolicy,
    base_dir: Path,
) -> PreparedStep:
    source = (base_dir / step.poscar).resolve()
    source_bytes = source.read_bytes()
    structure = _read_structure(source_bytes)
    _cell(structure)
    if structure.velocities is not None and any(
        x != 0 for row in structure.velocities for x in row
    ):
        raise ValueError(
            "VELOCITIES_UNSUPPORTED: v1 packages support only absent or all-zero velocities"
        )
    plan = step.potential_plan or select_by_policy(
        CandidateComposition.from_structure(structure, step.candidate_id),
        step.incar.goal,
        registry,
        policy,
    )
    incar_build = build_incar(structure, step.incar, plan, registry)
    data = incar_build.spec.model_dump(exclude_none=True)
    data.update(step.incar_options.model_dump(exclude_unset=True))
    incar = IncarSpec.model_validate(data)
    if incar.ISTART not in {0, 1} or incar.ICHARG not in {0, 1, 2, 11}:
        raise ValueError(
            "RESTART_MODE: v1 supports ISTART=0/1 and ICHARG=0/1/2/11 only"
        )
    if incar.ICHARG == 0 and incar.ISTART != 1:
        raise ValueError(
            "RESTART_MODE: ICHARG=0 requires explicit ISTART=1 and a WAVECAR dependency"
        )
    if incar.SOCSCALE is not None and not workflow.capabilities.socscale:
        raise ValueError(
            "SOCSCALE_CAPABILITY: declare verified target-build support and evidence before generating SOCSCALE"
        )
    validation = validate_incar(
        incar, structure, goal=step.incar.goal, plan=plan, registry=registry
    )
    validation.raise_for_errors()
    kbuild = build_kpoints(structure, step.kpoints, incar, incar_goal=step.incar.goal)
    # Replace the context-free nested report with the full POTCAR-aware validation.
    kbuild.validation.incar_validation = validation
    required = set(validation.required_artifacts)
    declared = {d.artifact for d in step.dependencies}
    if required != declared:
        raise ValueError(
            f"ARTIFACT_BINDING: {step.id} requires {sorted(required)}, declares {sorted(declared)}"
        )
    potcar, potcar_manifest = prepare_potcar(plan, registry)
    payloads = {
        "POSCAR": render_poscar(structure).encode("utf-8"),
        "POTCAR": potcar,
        "INCAR": render_incar(incar).encode("utf-8"),
        "KPOINTS": render_kpoints(kbuild.spec).encode("utf-8"),
    }
    if parse_incar(payloads["INCAR"].decode()).spec != incar:
        raise ValueError(
            "INCAR_ROUNDTRIP: generated parameters changed during rendering"
        )
    if parse_kpoints(payloads["KPOINTS"].decode()).spec != kbuild.spec:
        raise ValueError(
            "KPOINTS_ROUNDTRIP: generated sampling changed during rendering"
        )
    status = StepStatus(
        id=step.id,
        input_state="waiting_for_upstream" if required else "prepared",
        dependencies=[PendingArtifact(**d.model_dump()) for d in step.dependencies],
        executable=incar.executable,
    )
    moments = incar.MAGMOM
    stride = 3 if incar.spinor else 1
    atom_mapping = [
        {
            "poscar_index": i + 1,
            "species": element,
            "magmom": moments[i * stride : (i + 1) * stride],
        }
        for i, element in enumerate(structure.site_species)
    ]
    manifest = {
        "schema_version": 1,
        "candidate_id": step.candidate_id,
        "status": status.model_dump(mode="json"),
        "goal": step.incar.goal.model_dump(mode="json"),
        "request": step.model_dump(mode="json", exclude_none=True),
        "poscar_source": {"path": step.poscar, "sha256": digest(source_bytes)},
        "files": {name: digest(payload) for name, payload in payloads.items()},
        "potential_plan": plan.model_dump(mode="json"),
        "potcar": potcar_manifest.model_dump(mode="json"),
        "incar": incar.model_dump(mode="json", exclude_none=True),
        "recipes": {
            "incar": incar_build.recipe,
            "incar_version": incar_build.recipe_version,
            "kpoints": kbuild.recipe,
            "kpoints_version": kbuild.recipe_version,
        },
        "moment_basis": incar_build.moment_basis,
        "atom_mapping": atom_mapping,
        "validation": kbuild.validation.model_dump(mode="json", exclude_none=True),
    }
    return PreparedStep(step, structure, incar, kbuild.spec, payloads, manifest)


def _check_dependency(
    source: PreparedStep, target: PreparedStep, artifact: str
) -> None:
    edge = f"{source.request.id} -> {target.request.id} ({artifact})"
    if not _same_structure(source, target):
        raise ValueError(
            f"ARTIFACT_STRUCTURE: {edge} requires identical ordered sites and physical cell"
        )
    if source.manifest["files"]["POTCAR"] != target.manifest["files"]["POTCAR"]:
        raise ValueError(f"ARTIFACT_POTCAR: {edge} uses different licensed potentials")
    a, b = source.incar, target.incar
    if (a.NSW or 0) != 0 or a.ICHARG not in {0, 1, 2}:
        raise ValueError(
            f"ARTIFACT_PRODUCER: {edge} must originate in a stationary self-consistent step"
        )
    flag = "LCHARG" if artifact == "CHGCAR" else "LWAVE"
    if getattr(a, flag) is not True:
        raise ValueError(
            f"ARTIFACT_OUTPUT_DISABLED: {edge} requires producer {flag}=True"
        )
    common = (
        "ENCUT",
        "PREC",
        "LREAL",
        "LASPH",
        "LSORBIT",
        "LNONCOLLINEAR",
        "ISPIN",
        "MAGMOM",
        "SAXIS",
        "SOCSCALE",
        "LDAU",
        "LDAUTYPE",
        "LDAUL",
        "LDAUU",
        "LDAUJ",
        "LMAXMIX",
    )
    for tag in common:
        if getattr(a, tag) != getattr(b, tag):
            raise ValueError(
                f"ARTIFACT_COMPATIBILITY: {edge} differs in {tag}; v1 uses conservative restart rules"
            )
    if a.LDAU and (a.LMAXMIX or 2) < 2 * max(a.LDAUL):
        raise ValueError(
            f"ARTIFACT_LMAXMIX: {edge} lacks correlated-shell charge information"
        )
    if artifact == "WAVECAR":
        if a.NBANDS is None or a.NBANDS != b.NBANDS:
            raise ValueError(
                f"WAVECAR_NBANDS: {edge} requires explicit identical NBANDS"
            )
        if a.ISYM != b.ISYM or _sampling(source.kpoints) != _sampling(target.kpoints):
            raise ValueError(
                f"WAVECAR_KPOINTS: {edge} requires identical sampling and symmetry settings"
            )
        if (
            source.kpoints.coordinate_mode == "cartesian"
            and source.structure.scale != target.structure.scale
        ):
            raise ValueError(
                f"WAVECAR_KPOINTS: {edge} changes the Cartesian k-point scale convention"
            )


def _check_scans(steps: dict[str, PreparedStep], groups: list[dict]) -> None:
    for group in groups:
        baseline = steps[group["steps"][0]]
        varied = group["varied_tag"]
        for name in group["steps"]:
            step = steps[name]
            if (
                step.incar.ICHARG != 2
                or step.incar.ISTART != 0
                or step.request.dependencies
            ):
                raise ValueError(
                    "SCAN_METHOD: v1 scans require independent self-consistent starts"
                )
            if (
                not _same_structure(baseline, step)
                or baseline.payloads["POTCAR"] != step.payloads["POTCAR"]
            ):
                raise ValueError(
                    "SCAN_INPUTS: all variants require identical structure and potentials"
                )
            if _sampling(step.kpoints) != _sampling(baseline.kpoints):
                raise ValueError(
                    "SCAN_KPOINTS: all variants require identical sampling"
                )
            if step.incar.model_dump(exclude={varied}) != baseline.incar.model_dump(
                exclude={varied}
            ):
                raise ValueError(f"SCAN_INCAR: variants may differ only in {varied}")
            if group["kind"] == "mae":
                moments = np.array(step.incar.MAGMOM).reshape(-1, 3)
                if np.any(moments[:, :2] != 0) or not np.any(moments[:, 2] != 0):
                    raise ValueError(
                        "MAE_MOMENTS: SAXIS-only scan requires nonzero collinear moments along spin-frame z"
                    )


def prepare_workflow(
    workflow: Workflow,
    registry: PotentialRegistry,
    policy: PotentialPolicy,
    base_dir: Path,
    *,
    sources: dict | None = None,
) -> PreparedWorkflow:
    # JSON revalidation also checks mutated nested models and restores field-set semantics.
    workflow = Workflow.model_validate_json(
        workflow.model_dump_json(exclude_unset=True)
    )
    registry = PotentialRegistry.model_validate_json(registry.model_dump_json())
    policy = PotentialPolicy.model_validate_json(policy.model_dump_json())
    names = [record.potential for record in registry.records]
    if len(set(names)) != len(names):
        raise ValueError(
            "DUPLICATE_POTENTIAL: registry potential names must be unambiguous"
        )
    steps, groups = expand_steps(workflow)
    requests = {step.id: step for step in steps}
    graph = {step.id: [d.from_step for d in step.dependencies] for step in steps}
    for target, parents in graph.items():
        for source in parents:
            if source not in requests:
                raise ValueError(
                    f"MISSING_STEP: {target} references unknown step {source}"
                )
    try:
        order = list(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise ValueError("DEPENDENCY_CYCLE: workflow must be acyclic") from exc
    prepared = {}
    for name in order:
        try:
            prepared[name] = _compile_step(
                requests[name], workflow, registry, policy, base_dir
            )
        except (ValueError, OSError) as exc:
            raise ValueError(f"step {name}: {exc}") from exc
        for dep in requests[name].dependencies:
            _check_dependency(prepared[dep.from_step], prepared[name], dep.artifact)
    _check_scans(prepared, groups)
    manifest = {
        "schema_version": 1,
        "workflow_id": workflow.workflow_id,
        "mode": "local_dry_run",
        "generator_version": __version__,
        "recipe_version": "workflow/1",
        "validation": {
            "valid": True,
            "scope": "inputs_and_declared_dependencies",
            "runtime_verified": False,
        },
        "specification": workflow.model_dump(mode="json", exclude_unset=True),
        "provenance": {
            "runtime": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("pymatgen", "numpy", "pydantic")},
            },
            "sources": sources or {},
            "registry_model_sha256": digest(
                json_bytes(registry.model_dump(mode="json"))
            ),
            "policy_model_sha256": digest(json_bytes(policy.model_dump(mode="json"))),
        },
        "order": order,
        "dependency_graph": graph,
        "comparisons": groups,
        "steps": {
            name: {
                "status": step.manifest["status"],
                "files": step.manifest["files"],
                "manifest_sha256": digest(json_bytes(step.manifest)),
            }
            for name, step in prepared.items()
        },
        "runtime_requirements": [
            "Verify VASP executable/version and declared extensions",
            "Verify convergence and actual CHGCAR/WAVECAR compatibility before releasing dependents",
            "Record actual artifact hashes at execution time; no future artifact hash is invented",
        ],
    }
    return PreparedWorkflow(prepared, manifest)


def materialize_workflow(prepared: PreparedWorkflow, output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output exists; choose a new workflow directory")
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    # All writes stay inside this owned staging directory until the complete bundle is published.
    with tempfile.TemporaryDirectory(
        prefix=".magvasp-stage-", dir=output.parent
    ) as scratch:
        bundle = Path(scratch) / "bundle"
        bundle.mkdir()
        for name, step in prepared.steps.items():
            folder = bundle / safe_id(name)
            folder.mkdir()
            for filename in INPUT_NAMES:
                payload = step.payloads[filename]
                if digest(payload) != step.manifest["files"][filename]:
                    raise ValueError("Prepared input changed before materialization")
                (folder / filename).write_bytes(payload)
            manifest_bytes = json_bytes(step.manifest)
            if (
                digest(manifest_bytes)
                != prepared.manifest["steps"][name]["manifest_sha256"]
            ):
                raise ValueError(
                    "Prepared step manifest changed before materialization"
                )
            (folder / "manifest.json").write_bytes(manifest_bytes)
        (bundle / "workflow.manifest.json").write_bytes(json_bytes(prepared.manifest))
        inspect_bundle(bundle)
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                "Output appeared during generation; refusing to overwrite"
            )
        # On the supported Windows host rename also refuses an existing destination.
        os.rename(bundle, output)
    return output / "workflow.manifest.json"


def inspect_bundle(root: Path) -> dict:
    """Check package integrity only; never infer runtime convergence or release dependencies."""
    root = root.resolve()
    manifest = json.loads((root / "workflow.manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("steps"), dict)
    ):
        raise ValueError("Unsupported workflow manifest")
    if (
        not manifest["steps"]
        or len(manifest.get("order", [])) != len(manifest["steps"])
        or set(manifest.get("order", [])) != set(manifest["steps"])
    ):
        raise ValueError("Manifest order must cover all steps")
    for name, record in manifest["steps"].items():
        folder = root / safe_id(name)
        if folder.is_symlink() or folder.resolve().parent != root:
            raise ValueError("Step directory must remain inside the package")
        if set(record["files"]) != set(INPUT_NAMES):
            raise ValueError("Manifest must cover exactly four VASP inputs")
        expected = {**record["files"], "manifest.json": record["manifest_sha256"]}
        for filename, sha in expected.items():
            path = folder / filename
            if path.is_symlink() or path.resolve().parent != folder.resolve():
                raise ValueError("Package files must not be links outside their step")
            if digest(path.read_bytes()) != sha:
                raise ValueError(f"HASH_MISMATCH: {name}/{filename}")
        step_manifest = json.loads(
            (folder / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            step_manifest["files"] != record["files"]
            or step_manifest["status"] != record["status"]
        ):
            raise ValueError("Step and workflow manifests disagree")
    return {
        "integrity_valid": True,
        "runtime_verified": False,
        "workflow_id": manifest["workflow_id"],
        "steps": len(manifest["steps"]),
        "waiting_for_upstream": [
            name
            for name, item in manifest["steps"].items()
            if item["status"]["input_state"] == "waiting_for_upstream"
        ],
    }
