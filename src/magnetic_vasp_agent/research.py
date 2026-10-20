"""mCIF-first research planning and reproducible magnetic workflows.

This module is deliberately separate from the low-level VASP input compiler. It
turns one or more magnetic CIF candidates into a reviewed research bundle. The
supported stages produce ordinary ``Workflow`` requests; stages that require a
different solver are represented as explicit, non-executable plan items.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from magnetic_vasp_agent.incar_models import HubbardSpec, IncarRecipeRequest
from magnetic_vasp_agent.mcif import map_moments, read_mcif
from magnetic_vasp_agent.mcif_models import CandidateStructure
from magnetic_vasp_agent.models import CalculationGoal, CalculationTask, MagneticMode
from magnetic_vasp_agent.poscar import render_poscar
from magnetic_vasp_agent.workflow_models import (
    ArtifactDependency,
    CalculationStep,
    Workflow,
    safe_id,
)

ResearchTask = Literal[
    "magnetic_order",
    "relax",
    "static",
    "dos",
    "band",
    "mae",
    "convergence",
    "hubbard_sensitivity",
    "exchange",
    "finite_temperature",
    "spin_dynamics",
    "aimd",
    "pdos",
    "phonon",
    "elastic",
    "strain",
    "defect",
    "dmi",
    "neb",
    "wannier",
    "berry_transport",
    "hse",
    "gw",
    "dmft",
]


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tasks: list[ResearchTask] = Field(
        default_factory=lambda: ["magnetic_order", "relax", "static", "dos", "mae", "convergence"]
    )
    magnetic_mode: Literal["auto", "collinear", "noncollinear", "soc"] = "auto"
    encut: float = Field(default=600.0, gt=0)
    kpoint_spacing: float = Field(default=0.25, gt=0)
    dos_spacing: float = Field(default=0.15, gt=0)
    encut_values: list[float] = Field(default_factory=lambda: [500.0, 600.0, 700.0])
    kpoint_spacings: list[float] = Field(default_factory=lambda: [0.30, 0.25, 0.20])
    mae_axes: dict[str, list[float]] = Field(
        default_factory=lambda: {"001": [0.0, 0.0, 1.0], "100": [1.0, 0.0, 0.0], "110": [1.0, 1.0, 0.0]}
    )
    max_magnetic_variants: int = Field(default=8, ge=1, le=128)
    include_external_tasks: bool = True
    hubbard_sets: dict[str, dict[str, HubbardSpec]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_values(self) -> "ResearchConfig":
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be a nonempty list without duplicates")
        static_sources = {"static", "magnetic_order"}
        if ({"dos", "band", "pdos"} & set(self.tasks)) and not (static_sources & set(self.tasks)):
            raise ValueError("DOS/PDOS/Band requires an explicit static or magnetic_order stage")
        if "convergence" in self.tasks and len(self.encut_values) < 2 and len(self.kpoint_spacings) < 2:
            raise ValueError("convergence requires at least two ENCUT or k-point values")
        for label, axis in self.mae_axes.items():
            if not safe_id(label) or len(axis) != 3 or not any(axis):
                raise ValueError(f"invalid MAE axis {label!r}")
        if any(value <= 0 for value in self.encut_values + self.kpoint_spacings):
            raise ValueError("convergence values must be positive")
        return self

    @model_validator(mode="after")
    def validate_hubbard_sets(self) -> "ResearchConfig":
        if self.hubbard_sets and "hubbard_sensitivity" not in self.tasks:
            raise ValueError("hubbard_sets require task='hubbard_sensitivity'")
        for label, elements in self.hubbard_sets.items():
            safe_id(label)
            if not elements:
                raise ValueError("each Hubbard set must include at least one element")
        return self


ALL_RESEARCH_TASKS: tuple[ResearchTask, ...] = (
    "magnetic_order", "relax", "static", "dos", "band", "mae", "convergence",
    "hubbard_sensitivity", "exchange", "finite_temperature", "spin_dynamics", "aimd",
    "pdos", "phonon", "elastic", "strain", "defect", "dmi", "neb", "wannier",
    "berry_transport", "hse", "gw", "dmft",
)


class ResearchStage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task: ResearchTask
    status: Literal["workflow_generated", "requires_external_backend", "requires_manual_input"]
    description: str
    expected_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    plan_id: str
    sources: list[dict]
    candidates: list[dict]
    stages: list[ResearchStage]
    workflow_request: str | None = None
    generated_files: list[str] = Field(default_factory=list)
    generated_sha256: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    scientific_converged: Literal[False] = False


_EXTERNAL: dict[ResearchTask, tuple[str, list[str], list[str], list[str]]] = {
    "exchange": (
        "Fit Heisenberg exchange parameters from multiple converged magnetic states.",
        ["converged magnetic-order energies", "spin configurations", "unit-cell mapping"],
        ["J_ij fit", "fit residual", "supercell sensitivity"],
        ["common structure/XC/U/J/POTCAR", "energy uncertainty below fit signal"],
    ),
    "finite_temperature": (
        "Map J_ij and anisotropy to Monte Carlo, RPA, or atomistic spin dynamics.",
        ["J_ij", "MAE/DMI where applicable"],
        ["Tc/TN", "M(T)", "phase-transition diagnostics"],
        ["finite-size and sampling convergence", "model uncertainty"],
    ),
    "spin_dynamics": (
        "Simulate finite-temperature spin dynamics from a validated spin Hamiltonian.",
        ["J_ij", "anisotropy", "DMI if relevant", "spin moments"],
        ["M(T)", "dynamical correlations", "transition diagnostics"],
        ["time-step, system-size, and sampling convergence"],
    ),
    "aimd": (
        "Run ab initio molecular dynamics to sample structural/magnetic thermal stability.",
        ["relaxed magnetic structure", "temperature/ensemble protocol"],
        ["trajectory", "temperature and moment statistics"],
        ["time-step, cell-size, and sampling convergence"],
    ),
    "pdos": (
        "Project the converged density of states onto atoms and orbitals.",
        ["converged static charge density", "projection settings"],
        ["atom/orbital-resolved DOS"],
        ["dense k mesh and energy-window convergence"],
    ),
    "phonon": (
        "Run phonopy or DFPT force calculations for dynamical stability.",
        ["relaxed structure", "forces or displaced supercells"],
        ["phonon dispersion", "imaginary-mode report"],
        ["supercell and displacement convergence"],
    ),
    "elastic": (
        "Calculate finite-strain stresses or elastic constants.",
        ["relaxed structure", "strain set"],
        ["elastic tensor", "stability criteria"],
        ["symmetry and strain-amplitude convergence"],
    ),
    "strain": (
        "Scan strain or pressure and repeat magnetic-state comparisons.",
        ["strain tensor/volume grid", "magnetic candidates"],
        ["energy, moment, and MAE versus strain"],
        ["same comparison protocol at every point"],
    ),
    "defect": (
        "Build vacancy, antisite, or substitution structures and formation energies.",
        ["defect structure", "chemical potentials", "charge states if relevant"],
        ["formation energy", "relaxed moments", "charge correction"],
        ["cell-size and correction convergence"],
    ),
    "dmi": (
        "Evaluate Dzyaloshinskii-Moriya interactions with SOC/noncollinear methods.",
        ["SOC magnetic states", "q points or constrained rotations"],
        ["DMI vectors", "chirality convention"],
        ["SOC/k-point and rotation-angle convergence"],
    ),
    "neb": (
        "Calculate a magnetic or structural switching barrier with NEB images.",
        ["initial/final states", "intermediate images"],
        ["barrier", "image convergence", "moment continuity"],
        ["image count and spring-force convergence"],
    ),
    "wannier": (
        "Construct and validate a Wannier90 tight-binding model.",
        ["uniform SOC bands", "projections", "disentanglement window"],
        ["band-interpolation error", "real-space Hamiltonian"],
        ["uniform-grid and window convergence"],
    ),
    "berry_transport": (
        "Integrate Berry curvature for anomalous/spin Hall or related responses.",
        ["validated Wannier model", "dense full-BZ mesh", "SOC"],
        ["AHC/SHC/BCD", "mesh convergence"],
        ["gauge/interpolation and BZ convergence"],
    ),
    "hse": (
        "Use a screened hybrid functional for a sensitivity or final electronic structure.",
        ["converged semilocal state", "exact-exchange settings"],
        ["energy, moments, gap, bands"],
        ["k-point, empty-band, and mixing convergence"],
    ),
    "gw": (
        "Calculate quasiparticle energies with GW after a converged reference state.",
        ["reference WAVECAR/CHGCAR", "empty bands"],
        ["quasiparticle gap/bands"],
        ["empty-band, frequency, and k-point convergence"],
    ),
    "dmft": (
        "Use DFT+DMFT for temperature-dependent local correlation physics.",
        ["projected correlated subspace", "interaction parameters", "bath/solver settings"],
        ["spectral function", "moment versus temperature"],
        ["solver and interaction sensitivity"],
    ),
}


def _normal_axis(axis: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in axis))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("MAE axis must be finite and nonzero")
    return [round(value / norm, 12) for value in axis]


def _mode(candidate: CandidateStructure, requested: str) -> MagneticMode:
    if requested == "auto":
        return MagneticMode.COLLINEAR if candidate.collinear else MagneticMode.NONCOLLINEAR
    mode = MagneticMode(requested)
    if mode == MagneticMode.COLLINEAR and not candidate.collinear:
        raise ValueError(f"candidate {candidate.candidate_id} contains transverse moments")
    return mode


def _signed_magnitudes(candidate: CandidateStructure) -> list[float]:
    if not candidate.collinear or candidate.collinear_axis_cartesian is None:
        raise ValueError("magnetic-order and mCIF MAE variants currently require collinear moments")
    axis = candidate.collinear_axis_cartesian
    axis_norm = math.sqrt(sum(value * value for value in axis))
    unit = [value / axis_norm for value in axis]
    signed = []
    for site in candidate.sites:
        dot = sum(a * b for a, b in zip(site.moment_cartesian, unit, strict=True))
        signed.append(math.copysign(math.sqrt(sum(x * x for x in site.moment_cartesian)), dot) if dot else 0.0)
    return signed


def _explicit_request(candidate: CandidateStructure, goal: CalculationGoal, values: list[float], *, encut: float = 600.0, saxis: list[float] | None = None, hubbard: dict | None = None) -> IncarRecipeRequest:
    if goal.magnetic_mode == MagneticMode.COLLINEAR:
        magmom = values
    else:
        # Non-collinear mCIF moments already arrive as flattened Cartesian
        # triples.  Collinear candidates use signed magnitudes and need a
        # spin-frame z triple only for SOC/MAE scans.
        if len(values) == 3 * candidate.structure.num_atoms:
            magmom = values
        else:
            magmom = [component for value in values for component in (0.0, 0.0, value)]
    return IncarRecipeRequest(
        goal=goal,
        encut=encut,
        magmom=magmom,
        saxis=saxis or [0.0, 0.0, 1.0],
        hubbard=hubbard or {},
    )


def _initial_values(candidate: CandidateStructure, mode: MagneticMode) -> tuple[list[float], list[float]]:
    """Return scalar collinear moments and VASP spin-frame triples."""

    if candidate.collinear:
        signed = _signed_magnitudes(candidate)
        return signed, signed
    from magnetic_vasp_agent.mcif_models import McifIncarRequest

    request = McifIncarRequest(
        goal=CalculationGoal(task=CalculationTask.STATIC, magnetic_mode=mode),
        encut=600.0,
        moment_source="mcif",
        moment_mode="preserve_cartesian",
        saxis=[0.0, 0.0, 1.0],
    )
    mapped, _ = map_moments(candidate, request)
    return [], list(mapped.magmom)


def _variant_values(values: list[float], limit: int) -> list[tuple[str, list[float]]]:
    nonzero = [index for index, value in enumerate(values) if abs(value) > 1e-12]
    if not nonzero:
        return [("source", values)]
    variants: list[tuple[str, list[float]]] = [("source", values)]
    for bits in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        candidate = list(values)
        for index, sign in zip(nonzero, bits, strict=True):
            candidate[index] = abs(candidate[index]) * sign
        label = "m" + "".join("p" if bit > 0 else "n" for bit in bits)
        if candidate != values and all(existing[1] != candidate for existing in variants):
            variants.append((label, candidate))
        if len(variants) >= limit:
            break
    return variants


def _relative(target: Path, parent: Path) -> str:
    return Path(os.path.relpath(target.resolve(), parent.resolve())).as_posix()


def _candidate_stem(candidate_id: str) -> str:
    ascii_id = re.sub(r"[^A-Za-z0-9_-]+", "-", candidate_id).strip("-_") or "candidate"
    if len(ascii_id) > 72:
        suffix = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
        ascii_id = f"{ascii_id[:72]}-{suffix}"
    return safe_id(ascii_id)


def plan_research_from_mcif(
    mcif_paths: list[Path],
    output: Path,
    *,
    plan_id: str | None = None,
    config: ResearchConfig | None = None,
    notes: str | None = None,
) -> ResearchPlan:
    """Build a mCIF research bundle without running VASP or external solvers."""

    if not mcif_paths:
        raise ValueError("at least one mCIF path is required")
    if output.exists():
        raise FileExistsError("research output exists; choose a new path")
    cfg = config or ResearchConfig()
    candidates = [read_mcif(path) for path in mcif_paths]
    stems = [_candidate_stem(candidate.candidate_id) for candidate in candidates]
    if len({stem.casefold() for stem in stems}) != len(stems):
        raise ValueError("mCIF candidates need unique IDs for one research bundle")
    output.mkdir(parents=True)
    parent = output.resolve()
    workflow_steps: list[CalculationStep] = []
    stages: list[ResearchStage] = []
    generated: list[str] = []
    warnings: list[str] = []

    for candidate_index, candidate in enumerate(candidates):
        mode = _mode(candidate, cfg.magnetic_mode)
        source_path = Path(mcif_paths[candidate_index])
        source_rel = _relative(source_path, parent)
        signed, initial_values = _initial_values(candidate, mode)
        stem = stems[candidate_index]
        variants = _variant_values(signed, cfg.max_magnetic_variants) if "magnetic_order" in cfg.tasks and candidate.collinear else [("source", initial_values)]
        if "magnetic_order" in cfg.tasks and not candidate.collinear:
            warnings.append(f"{candidate.candidate_id}: noncollinear magnetic-order enumeration requires explicit texture candidates")
        variant_poscars: dict[str, Path] = {}
        if any(task in cfg.tasks for task in ("magnetic_order", "static", "dos", "convergence", "hubbard_sensitivity")):
            for label, _ in variants:
                poscar_path = output / f"{stem}-{label}.POSCAR"
                poscar_path.write_text(render_poscar(candidate.structure), encoding="utf-8", newline="\n")
                generated.append(str(poscar_path.name))
                variant_poscars[label] = poscar_path

        if "relax" in cfg.tasks:
            from magnetic_vasp_agent.mcif_models import McifIncarRequest

            goal = CalculationGoal(task=CalculationTask.RELAX, magnetic_mode=mode, notes=notes)
            for label, values in variants:
                if "magnetic_order" in cfg.tasks:
                    poscar_path = variant_poscars[label]
                    workflow_steps.append(CalculationStep(
                        id=f"relax-{stem}-{label}", candidate_id=f"{candidate.candidate_id}:{label}", poscar=poscar_path.name,
                        incar=_explicit_request(candidate, goal, values, encut=cfg.encut),
                        kpoints={"goal": goal.model_dump(mode="json"), "spacing": cfg.kpoint_spacing, "centering": "gamma"},
                    ))
                else:
                    workflow_steps.append(CalculationStep(
                        id=f"relax-{stem}", candidate_id=candidate.candidate_id, mcif=source_rel,
                        incar=McifIncarRequest(goal=goal, encut=cfg.encut, moment_source="mcif", moment_mode="preserve_cartesian" if mode != MagneticMode.COLLINEAR else "collinear_projection"),
                        kpoints={"goal": goal.model_dump(mode="json"), "spacing": cfg.kpoint_spacing, "centering": "gamma"},
                    ))
                    break
            warnings.append(f"{candidate.candidate_id}: re-import each CONTCAR and regenerate corresponding static/DOS steps after relax; no implicit CONTCAR dependency is created")

        source_static_id: str | None = None
        source_poscar: Path | None = None
        source_incar: IncarRecipeRequest | None = None
        source_goal: CalculationGoal | None = None
        for label, values in variants:
            poscar_path = variant_poscars.get(label)
            goal = CalculationGoal(task=CalculationTask.STATIC, magnetic_mode=mode, notes=notes)
            incar = _explicit_request(candidate, goal, values, encut=cfg.encut)
            static_id = f"static-{stem}-{label}"
            if "static" in cfg.tasks or "magnetic_order" in cfg.tasks:
                workflow_steps.append(CalculationStep(
                    id=static_id, candidate_id=f"{candidate.candidate_id}:{label}", poscar=poscar_path.name,
                    incar=incar, kpoints={"goal": goal.model_dump(mode="json"), "spacing": cfg.kpoint_spacing, "centering": "gamma"},
                ))
            if label == "source":
                source_static_id, source_poscar, source_incar, source_goal = static_id, poscar_path, incar, goal
        if "dos" in cfg.tasks and source_static_id and source_poscar and source_incar and source_goal:
            dos_goal = source_goal.model_copy(update={"task": CalculationTask.DOS})
            workflow_steps.append(CalculationStep(
                id=f"dos-{stem}", candidate_id=f"{candidate.candidate_id}:source", poscar=source_poscar.name,
                incar=source_incar.model_copy(update={"goal": dos_goal}),
                kpoints={"goal": dos_goal.model_dump(mode="json"), "spacing": cfg.dos_spacing, "centering": "gamma"},
                dependencies=[ArtifactDependency(artifact="CHGCAR", from_step=source_static_id)],
            ))

        if "hubbard_sensitivity" in cfg.tasks:
            goal = CalculationGoal(task=CalculationTask.STATIC, magnetic_mode=mode, notes=notes)
            poscar_path = variant_poscars.get("source")
            for label, shells in cfg.hubbard_sets.items():
                incar = _explicit_request(candidate, goal, initial_values, encut=cfg.encut, hubbard=shells)
                workflow_steps.append(CalculationStep(
                    id=f"hubbard-{stem}-{safe_id(label)}",
                    candidate_id=f"{candidate.candidate_id}:hubbard:{label}",
                    poscar=poscar_path.name,
                    incar=incar,
                    kpoints={"goal": goal.model_dump(mode="json"), "spacing": cfg.kpoint_spacing, "centering": "gamma"},
                ))

        if "mae" in cfg.tasks:
            if not candidate.collinear:
                warnings.append(f"{candidate.candidate_id}: MAE directions require explicit rotation for a noncollinear texture; stage is manual")
            else:
                for label, axis in cfg.mae_axes.items():
                    axis = _normal_axis(axis)
                    poscar_path = output / f"{stem}-mae-{safe_id(label)}.POSCAR"
                    poscar_path.write_text(render_poscar(candidate.structure), encoding="utf-8", newline="\n")
                    generated.append(str(poscar_path.name))
                    goal = CalculationGoal(task=CalculationTask.MAGNETIC_ANISOTROPY, magnetic_mode=MagneticMode.SOC, notes=notes)
                    workflow_steps.append(CalculationStep(
                        id=f"mae-{stem}-{safe_id(label)}", candidate_id=f"{candidate.candidate_id}:mae:{label}", poscar=poscar_path.name,
                        incar=_explicit_request(candidate, goal, signed, encut=cfg.encut, saxis=axis),
                        kpoints={"goal": goal.model_dump(mode="json"), "spacing": cfg.kpoint_spacing, "centering": "gamma"},
                    ))

        if "convergence" in cfg.tasks:
            goal = CalculationGoal(task=CalculationTask.STATIC, magnetic_mode=mode, notes=notes)
            values = signed if signed else initial_values
            poscar_path = variant_poscars.get("source")
            for index, encut in enumerate(cfg.encut_values):
                incar = _explicit_request(candidate, goal, values, encut=encut)
                workflow_steps.append(CalculationStep(
                    id=f"conv-{stem}-encut-{index+1}", candidate_id=f"{candidate.candidate_id}:conv:encut:{encut:g}", poscar=poscar_path.name,
                    incar=incar, kpoints={"goal": goal.model_dump(mode="json"), "spacing": cfg.kpoint_spacing, "centering": "gamma"},
                ))
            for index, spacing in enumerate(cfg.kpoint_spacings):
                incar = _explicit_request(candidate, goal, values, encut=cfg.encut)
                workflow_steps.append(CalculationStep(
                    id=f"conv-{stem}-kpoint-{index+1}", candidate_id=f"{candidate.candidate_id}:conv:kpoint:{spacing:g}", poscar=poscar_path.name,
                    incar=incar, kpoints={"goal": goal.model_dump(mode="json"), "spacing": spacing, "centering": "gamma"},
                ))

    # Only supported VASP stages are placed in the typed executable workflow.
    if "magnetic_order" in cfg.tasks:
        magnetic_steps = [step.id for step in workflow_steps if step.id.startswith("static-")]
        noncollinear = any(not candidate.collinear for candidate in candidates)
        stages.append(ResearchStage(task="magnetic_order", status="requires_manual_input" if noncollinear else "workflow_generated", description="Enumerate bounded collinear sign variants from mCIF moments; symmetry-equivalent candidates are not reduced. Noncollinear textures require explicit candidates.", expected_inputs=["mCIF"], expected_outputs=["independent static runs", "final local moments", "energy ranking"], acceptance=["same structure/POTCAR/XC/U/J", "energy_sigma0 or consistent energy definition", "detect moment collapse"], steps=magnetic_steps))
    if "relax" in cfg.tasks:
        stages.append(ResearchStage(task="relax", status="workflow_generated", description="Generate ionic relaxations for each selected magnetic start when enumeration is enabled.", expected_inputs=["mCIF or mapped magnetic POSCAR"], expected_outputs=["CONTCAR", "forces", "stress"], acceptance=["ionic convergence", "re-import CONTCAR before static"], steps=[step.id for step in workflow_steps if step.id.startswith("relax-")]))
    if "static" in cfg.tasks:
        stages.append(ResearchStage(task="static", status="workflow_generated", description="Run stationary calculations for the explicitly selected magnetic structures.", expected_inputs=["POSCAR", "POTCAR", "INCAR", "KPOINTS"], expected_outputs=["OUTCAR", "OSZICAR", "CHGCAR", "vasprun.xml"], acceptance=["electronic convergence", "final local moments"], steps=[step.id for step in workflow_steps if step.id.startswith("static-")]))
    for task in ("dos", "mae", "convergence", "pdos"):
        if task in cfg.tasks:
            if task == "dos":
                stage = ResearchStage(task="dos", status="workflow_generated", description="Compute DOS from a verified stationary CHGCAR.", expected_inputs=["converged static CHGCAR"], expected_outputs=["DOSCAR", "vasprun.xml"], acceptance=["CHGCAR hash and compatibility", "dense integration mesh"], steps=[step.id for step in workflow_steps if step.id.startswith("dos-")])
            elif task == "pdos":
                stage = ResearchStage(task="pdos", status="requires_external_backend", description="Project converged DOS onto atoms and orbitals using a versioned post-processing backend.", expected_inputs=["converged static CHGCAR", "projection settings"], expected_outputs=["atom/orbital-resolved DOS"], acceptance=["dense k mesh and energy-window convergence"], steps=[])
            elif task == "mae":
                mae_steps = [step.id for step in workflow_steps if step.id.startswith("mae-")]
                stage = ResearchStage(task="mae", status="workflow_generated" if mae_steps else "requires_manual_input", description="Compare self-consistent SOC energies for requested magnetization axes; noncollinear textures need an explicit physical rotation.", expected_inputs=["SOC POSCAR/INCAR", "identical k mesh and settings"], expected_outputs=["directional energies", "MAE table", "orbital moments"], acceptance=["energy and k-point convergence below MAE signal", "final moment direction check"], steps=mae_steps)
            else:
                stage = ResearchStage(task="convergence", status="workflow_generated", description="Generate independent ENCUT and k-point-spacing convergence calculations.", expected_inputs=["candidate POSCAR", "ENCUT grid", "k-point spacing grid"], expected_outputs=["energy/moment convergence table"], acceptance=["derived energy differences stable within target tolerance"], steps=[step.id for step in workflow_steps if step.id.startswith("conv-")])
            stages.append(stage)
    if "hubbard_sensitivity" in cfg.tasks:
        u_steps = [step.id for step in workflow_steps if step.id.startswith("hubbard-")]
        stages.append(ResearchStage(task="hubbard_sensitivity", status="workflow_generated" if u_steps else "requires_manual_input", description="Run only explicitly supplied DFT+U parameter sets; U/J values are not guessed.", expected_inputs=["reviewed element/l/U/J sets"], expected_outputs=["energy and moment versus U/J"], acceptance=["same structure, k mesh, functional and energy definition", "record U/J provenance"], steps=u_steps))
    if cfg.include_external_tasks:
        for task in cfg.tasks:
            if task in _EXTERNAL and task != "hubbard_sensitivity":
                description, inputs, outputs, acceptance = _EXTERNAL[task]
                stages.append(ResearchStage(task=task, status="requires_external_backend", description=description, expected_inputs=inputs, expected_outputs=outputs, acceptance=acceptance))
    for task in cfg.tasks:
        if task == "band":
            stages.append(ResearchStage(task="band", status="requires_manual_input", description="Band calculations require a validated high-symmetry path for the actual magnetic cell.", expected_inputs=["explicit line-mode path", "verified CHGCAR"], expected_outputs=["band energies", "spin-resolved plot"], acceptance=["path labels and cell basis verified"]))

    plan = ResearchPlan(
        plan_id=safe_id(plan_id or f"research-{stems[0]}"),
        sources=[{"path": str(path.resolve()), "candidate_id": candidate.candidate_id, "source_sha256": candidate.source_sha256} for path, candidate in zip(mcif_paths, candidates, strict=True)],
        candidates=[{"candidate_id": candidate.candidate_id, "declared_bns": candidate.declared_bns, "collinear": candidate.collinear, "species_order": candidate.structure.species_order, "counts": candidate.structure.counts, "total_moment_cartesian": candidate.total_moment_cartesian} for candidate in candidates],
        stages=stages,
        workflow_request="workflow.request.json" if workflow_steps else None,
        generated_files=generated,
        generated_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in generated},
        warnings=warnings,
    )
    if workflow_steps:
        workflow = Workflow(workflow_id=plan.plan_id, steps=workflow_steps)
        workflow_path = output / "workflow.request.json"
        workflow_path.write_text(json.dumps(workflow.model_dump(mode="json", exclude_unset=True), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        plan.generated_sha256[workflow_path.name] = hashlib.sha256(workflow_path.read_bytes()).hexdigest()
    (output / "research-plan.json").write_text(json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return plan

