"""Typed tool registry exposing the deterministic VASP core to an agent.

Each tool wraps an existing deterministic function. Tools never accept
arbitrary shell and never place POTCAR content in their result. Read-only tools
(``read_only``) parse or validate; local writes (``local_write``) generate files
under a caller-selected output path; elevated planning tools require approval.
Skill tools only read ``SKILL.md`` and text references. Cluster execution is
limited to fixed smoke, transfer, and launch operations; arbitrary launch,
cancel, retry, SSH, or MPI tools are not exposed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from magnetic_vasp_agent.agent.tool_models import (
    CONTROLLED_CLUSTER_EXECUTE_TOOLS,
    RiskTier,
    StrictModel,
    ToolSpec,
)
from magnetic_vasp_agent.execution import prepare_execution
from magnetic_vasp_agent.execution_approval import (
    MAX_GRANT_MINUTES,
    RESOURCE_GRANT_CONFIRM,
    issue_resource_grant,
    validate_human_approver,
)
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    ExecutionRequest,
    ExecutionState,
    ExecutionStateMachine,
    ExecutionStage,
    ProbeSnapshot,
    ResourceGrant,
)
from magnetic_vasp_agent.execution_probe import (
    assess_resources,
    collect_probe,
    probe_preview,
)
from magnetic_vasp_agent.execution_remote import run_mpi_smoke, write_json_report
from magnetic_vasp_agent.execution_monitor import fetch_results, remote_status
from magnetic_vasp_agent.execution_artifacts import release_chgcar
from magnetic_vasp_agent.execution_runtime import status as inspect_run
from magnetic_vasp_agent.execution_runtime import launch as launch_runtime
from magnetic_vasp_agent.execution_runtime_models import (
    REMOTE_CONFIRM,
    SMOKE_CONFIRM,
    LaunchBackend,
    LaunchRequest,
)
from magnetic_vasp_agent.execution_simulator import simulate
from magnetic_vasp_agent.incar import read_incar, render_incar, validate_incar, write_incar
from magnetic_vasp_agent.incar_models import IncarRecipeRequest
from magnetic_vasp_agent.incar_recipes import build_incar
from magnetic_vasp_agent.knowledge.retrieval import (
    build_answer_draft,
    build_answer_response,
    build_evidence_package,
    search_bm25,
    search_vector,
    write_answer_draft,
    write_evidence_package,
)
from magnetic_vasp_agent.knowledge.models import KnowledgeSourceType
from magnetic_vasp_agent.knowledge.retrieval_models import (
    AnswerRequest,
    EvidencePackage,
    RetrievalFilter,
)
from magnetic_vasp_agent.kpoints import (
    read_kpoints,
    render_kpoints,
    validate_kpoints,
    write_kpoints,
)
from magnetic_vasp_agent.kpoints_models import KpointsRecipeRequest
from magnetic_vasp_agent.kpoints_recipes import build_kpoints
from magnetic_vasp_agent.mcif import read_mcif
from magnetic_vasp_agent.input_review import build_review_card, load_review_history
from magnetic_vasp_agent.input_review_models import InputReviewCard
from magnetic_vasp_agent.mcif_workflow import plan_workflow_from_mcif
from magnetic_vasp_agent.models import (
    CalculationGoal,
    CalculationTask,
    CandidateComposition,
    MagneticMode,
    PotentialPlan,
    PotentialPolicy,
    PotentialRegistry,
)
from magnetic_vasp_agent.poscar import read_poscar
from magnetic_vasp_agent.postprocess import analyze_calculation
from magnetic_vasp_agent.vaspkit import run_vaspkit, run_vaspkit_task
from magnetic_vasp_agent.vaspkit_catalog import list_examples
from magnetic_vasp_agent.vaspkit_plot import plot_band, plot_tdos
from magnetic_vasp_agent.skills import (
    DEFAULT_SKILL_CATALOG,
    list_skills,
    read_skill_file,
    show_skill,
)
from magnetic_vasp_agent.potcar import (
    assemble_potcar,
    load_policy,
    load_registry,
    select_by_policy,
)
from magnetic_vasp_agent.transfer import (
    build_transfer_plan,
    transfer_bundle,
    write_transfer_report,
)
from magnetic_vasp_agent.transfer_models import TransferRequest
from magnetic_vasp_agent.workflow import digest, inspect_bundle, materialize_workflow, prepare_workflow
from magnetic_vasp_agent.workflow_models import ParameterScan, Workflow


@dataclass(frozen=True)
class Tool:
    """A registered tool: public spec, typed input model, and its runner."""

    spec: ToolSpec
    input_model: type[BaseModel]
    run: Callable[[BaseModel], dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        risk_tier: RiskTier,
        produces_files: bool,
        input_model: type[BaseModel],
        run: Callable[[BaseModel], dict[str, Any]],
    ) -> None:
        if (
            risk_tier == RiskTier.CLUSTER_EXECUTE
            and name not in CONTROLLED_CLUSTER_EXECUTE_TOOLS
        ):
            raise ValueError("only fixed controlled tools may use cluster_execute")
        if name in self._tools:
            raise ValueError(f"duplicate tool name: {name}")
        spec = ToolSpec(
            name=name,
            description=description,
            risk_tier=risk_tier,
            produces_files=produces_files,
            input_schema=input_model.model_json_schema(),
        )
        self._tools[name] = Tool(spec=spec, input_model=input_model, run=run)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            known = ", ".join(sorted(self._tools)) or "<none>"
            raise KeyError(f"unknown tool {name!r}; registered tools: {known}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in self.names()]


# --------------------------------------------------------------------------- #
# Input models. Enum-like fields are plain strings and converted inside the    #
# tool so JSON arguments validate cleanly under strict mode.                    #
# --------------------------------------------------------------------------- #


class StructureShowInput(StrictModel):
    poscar: str = Field(min_length=1)


class IncarShowInput(StrictModel):
    incar: str = Field(min_length=1)
    strict: bool = False


class IncarValidateInput(StrictModel):
    incar: str = Field(min_length=1)
    poscar: str = Field(min_length=1)
    plan: str | None = None
    registry: str | None = None
    executable: str | None = None
    strict: bool = False


class KpointsShowInput(StrictModel):
    kpoints: str = Field(min_length=1)


class KpointsValidateInput(StrictModel):
    kpoints: str = Field(min_length=1)
    poscar: str = Field(min_length=1)
    incar: str = Field(min_length=1)
    incar_request: str = Field(min_length=1)


class McifShowInput(StrictModel):
    mcif: str = Field(min_length=1)
    candidate_id: str | None = None


class WorkflowFromMcifInput(StrictModel):
    mcif: list[str] = Field(min_length=1)
    output: str = Field(min_length=1)
    workflow_id: str | None = None
    task: str = "static"
    magnetic_mode: str = "auto"
    encut: float = 600.0
    spacing: float = 0.25
    include_dos: bool = False
    dos_spacing: float = 0.15
    notes: str | None = None


class PotcarSelectInput(StrictModel):
    registry: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    poscar: str = Field(min_length=1)
    candidate_id: str | None = None
    task: str = "static"
    magnetic_mode: str = "collinear"


class WorkflowRequestInput(StrictModel):
    request: str = Field(min_length=1)
    registry: str = Field(min_length=1)
    policy: str = Field(min_length=1)


class WorkflowBuildInput(WorkflowRequestInput):
    output: str = Field(min_length=1)


class WorkflowMaeScanInput(WorkflowRequestInput):
    """Build independent SOC inputs for each requested magnetization axis."""

    axes: dict[str, list[float]] = Field(min_length=2)
    output: str = Field(min_length=1)


class WorkflowInspectInput(StrictModel):
    bundle: str = Field(min_length=1)


class WorkflowReviewCardInput(StrictModel):
    bundle: str = Field(min_length=1)


class WorkflowReviewHistoryInput(StrictModel):
    species: list[str] = Field(default_factory=list)
    task: str | None = None
    magnetic_mode: str | None = None
    reviewer: str | None = None
    limit: int = Field(default=5, ge=1, le=50)


class KnowledgeSearchInput(StrictModel):
    index: str = Field(min_length=1)
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)
    backend: str = "bm25"
    source_types: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    magnetic_modes: list[str] = Field(default_factory=list)
    validated_only: bool = False
    model_dir: str | None = None


class KnowledgeAnswerInput(StrictModel):
    index: str = Field(min_length=1)
    question: str = Field(min_length=1)
    claims: list[str] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=100)
    backend: str = "bm25"
    required_source_types: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    magnetic_modes: list[str] = Field(default_factory=list)
    validated_only: bool = False
    human_review: bool = False
    model_dir: str | None = None


class ExecutionProbeInput(StrictModel):
    profile: str = Field(min_length=1)
    node: str = Field(min_length=1)


class ExecutionProbeLiveInput(StrictModel):
    """Run the fixed batch-key read-only probe and persist its snapshot."""

    profile: str = Field(min_length=1)
    node: str = Field(min_length=1)
    output: str = Field(min_length=1)


class ExecutionPlanInput(StrictModel):
    bundle: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    request: str = Field(min_length=1)
    snapshot: str | None = None


class ExecutionRequestCreateInput(StrictModel):
    profile: str = Field(min_length=1)
    node: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    mpi_ranks: int = Field(default=1, ge=1, le=1024)
    cpu_ids: list[int] | None = None
    output: str = Field(min_length=1)


class ExecutionTransferRequestCreateInput(StrictModel):
    profile: str = Field(min_length=1)
    execution_request: str = Field(min_length=1)
    output: str = Field(min_length=1)

class ExecutionResourcePlanInput(StrictModel):
    profile: str = Field(min_length=1)
    request: str = Field(min_length=1)
    snapshot: str = Field(min_length=1)
    output: str = Field(min_length=1)


class ExecutionResourceGrantInput(StrictModel):
    bundle: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    request: str = Field(min_length=1)
    approver: str = Field(min_length=1)
    confirm: str = Field(min_length=1)
    probe_output: str = Field(min_length=1)
    output: str = Field(min_length=1)
    valid_minutes: int = Field(default=30, ge=1, le=MAX_GRANT_MINUTES)
    grant_id: str | None = None

class ExecutionTopssSnapshotInput(StrictModel):
    profile: str = Field(min_length=1)
    output: str = Field(min_length=1)


class ExecutionSmokeInput(StrictModel):
    profile: str = Field(min_length=1)
    request: str = Field(min_length=1)
    resource_grant: str = Field(min_length=1)
    confirm: str = Field(min_length=1)
    probe_output: str = Field(min_length=1)
    output: str = Field(min_length=1)


class ExecutionTransferInput(StrictModel):
    bundle: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    request: str = Field(min_length=1)
    resource_grant: str = Field(min_length=1)
    output: str = Field(min_length=1)


class ExecutionRemoteStatusInput(StrictModel):
    profile: str = Field(min_length=1)
    run: str = Field(min_length=1)
    output: str = Field(min_length=1)


class ExecutionFetchInput(StrictModel):
    profile: str = Field(min_length=1)
    run: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    files: list[str] = Field(min_length=1, max_length=32)
    output: str | None = None


class PotcarBuildInput(StrictModel):
    registry: str = Field(min_length=1)
    plan: str = Field(min_length=1)
    output: str = Field(min_length=1)


class IncarBuildInput(StrictModel):
    request: str = Field(min_length=1)
    poscar: str = Field(min_length=1)
    plan: str = Field(min_length=1)
    registry: str = Field(min_length=1)
    output: str = Field(min_length=1)


class KpointsBuildInput(StrictModel):
    request: str = Field(min_length=1)
    incar_request: str = Field(min_length=1)
    incar: str = Field(min_length=1)
    poscar: str = Field(min_length=1)
    output: str = Field(min_length=1)


class ExecutionSimulateInput(StrictModel):
    output: str = Field(min_length=1)
    scenario: str = "complete"


class ExecutionInspectRunInput(StrictModel):
    run: str = Field(min_length=1)


class SkillListInput(StrictModel):
    catalog: str = str(DEFAULT_SKILL_CATALOG)


class SkillShowInput(StrictModel):
    skill_id: str = Field(min_length=1)
    catalog: str = str(DEFAULT_SKILL_CATALOG)


class SkillReadInput(StrictModel):
    skill_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    catalog: str = str(DEFAULT_SKILL_CATALOG)


class ExecutionGateInspectInput(StrictModel):
    """Inspect a local execution-gate record; never contacts a cluster."""

    state: str = Field(min_length=1)


class KnowledgeEvidenceInput(StrictModel):
    index: str = Field(min_length=1)
    query: str = Field(min_length=1)
    output: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)
    backend: str = "bm25"
    required_source_types: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    magnetic_modes: list[str] = Field(default_factory=list)
    validated_only: bool = False
    human_review: bool = False
    model_dir: str | None = None


class KnowledgeDraftInput(StrictModel):
    evidence: str = Field(min_length=1)
    output: str = Field(min_length=1)
    claims: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ExecutionTransferPlanInput(StrictModel):
    bundle: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    request: str = Field(min_length=1)

class ExecutionLaunchInput(StrictModel):
    backend: str = "remote_mpi"
    confirm: str = Field(min_length=1)
    output: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    bundle: str = Field(min_length=1)
    request: str = Field(min_length=1)
    transfer_report: str = Field(min_length=1)
    smoke_report: str = Field(min_length=1)
    resource_grant: str = Field(min_length=1)
    probe_output: str = Field(min_length=1)
    artifact_release: str | None = None


class ExecutionArtifactReleaseInput(StrictModel):
    """Promote a verified SCF CHGCAR to one declared dependent step."""

    bundle: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    source_run: str = Field(min_length=1)
    source_status: str = Field(min_length=1)
    fetch_report: str = Field(min_length=1)
    target_transfer_report: str = Field(min_length=1)
    target_step: str = Field(min_length=1)
    confirm: str = Field(min_length=1)
    output: str = Field(min_length=1)


class PostprocessParseInput(StrictModel):
    directory: str = Field(min_length=1)

class VaspkitInput(StrictModel):
    directory: str = Field(min_length=1)
    task: str = Field(min_length=1)
    executable: str = Field(min_length=1)
    backend: str = "native"
    wsl_distribution: str | None = None

class VaspkitCatalogInput(StrictModel):
    root: str = Field(min_length=1)

class VaspkitTaskInput(StrictModel):
    directory: str = Field(min_length=1)
    task_id: int = Field(ge=0, le=9999)
    executable: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list, max_length=32)
    backend: str = "native"
    wsl_distribution: str | None = None


class VaspkitPlotDosInput(StrictModel):
    data: str = Field(min_length=1)
    output: str = Field(min_length=1)


class VaspkitPlotBandInput(StrictModel):
    data: str = Field(min_length=1)
    labels: str = Field(min_length=1)
    output: str = Field(min_length=1)


# --------------------------------------------------------------------------- #
# Small helpers                                                                 #
# --------------------------------------------------------------------------- #


def _enum_list(values: list[str], enum_type: type) -> list:
    try:
        return [enum_type(item) for item in values]
    except ValueError as exc:  # pragma: no cover - message tested via runner
        raise ValueError(f"invalid enum value in {values!r}: {exc}") from exc


def _refuse_existing(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output exists; choose a new path: {path}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _retrieval_response(
    index: str,
    query: str,
    filters: RetrievalFilter,
    limit: int,
    backend: str,
    model_dir: str | None,
):
    index_path = Path(index)
    if backend == "bm25":
        if model_dir is not None:
            raise ValueError("model_dir requires backend=vector")
        return search_bm25(index_path, query, filters, limit)
    if backend == "vector":
        return search_vector(
            index_path,
            query,
            filters,
            limit,
            model_dir=Path(model_dir) if model_dir else None,
        )
    raise ValueError("backend must be bm25 or vector")


# --------------------------------------------------------------------------- #
# Tool runners                                                                  #
# --------------------------------------------------------------------------- #


def _run_structure_show(data: StructureShowInput) -> dict[str, Any]:
    return read_poscar(Path(data.poscar)).model_dump(mode="json")


def _run_postprocess_parse(data: PostprocessParseInput) -> dict[str, Any]:
    return analyze_calculation(Path(data.directory)).model_dump(
        mode="json", exclude_none=True
    )

def _run_vaspkit(data: VaspkitInput) -> dict[str, Any]:
    return run_vaspkit(
        Path(data.directory),
        data.task,
        Path(data.executable),
        backend=data.backend,
        wsl_distribution=data.wsl_distribution,
    )

def _run_vaspkit_catalog(data: VaspkitCatalogInput) -> dict[str, Any]:
    return list_examples(Path(data.root))

def _run_vaspkit_task(data: VaspkitTaskInput) -> dict[str, Any]:
    return run_vaspkit_task(
        Path(data.directory),
        data.task_id,
        Path(data.executable),
        data.inputs,
        backend=data.backend,
        wsl_distribution=data.wsl_distribution,
    )


def _run_vaspkit_plot_dos(data: VaspkitPlotDosInput) -> dict[str, Any]:
    return plot_tdos(Path(data.data), Path(data.output)).model_dump(mode="json")


def _run_vaspkit_plot_band(data: VaspkitPlotBandInput) -> dict[str, Any]:
    return plot_band(
        Path(data.data), Path(data.labels), Path(data.output)
    ).model_dump(mode="json")


def _run_incar_show(data: IncarShowInput) -> dict[str, Any]:
    document = read_incar(Path(data.incar), legacy_booleans=not data.strict)
    return document.model_dump(mode="json", exclude_none=True)


def _run_incar_validate(data: IncarValidateInput) -> dict[str, Any]:
    if (data.plan is None) != (data.registry is None):
        raise ValueError("plan and registry must be supplied together")
    document = read_incar(Path(data.incar), legacy_booleans=not data.strict)
    plan = (
        PotentialPlan.model_validate_json(Path(data.plan).read_text(encoding="utf-8"))
        if data.plan
        else None
    )
    report = validate_incar(
        document.spec,
        read_poscar(Path(data.poscar)),
        plan=plan,
        registry=load_registry(Path(data.registry)) if data.registry else None,
        goal=plan.goal if plan else None,
        executable=data.executable,
        source_lines=document.source_lines,
    )
    report.diagnostics = document.diagnostics + report.diagnostics
    return report.model_dump(mode="json", exclude_none=True)


def _run_kpoints_show(data: KpointsShowInput) -> dict[str, Any]:
    return read_kpoints(Path(data.kpoints)).model_dump(mode="json", exclude_none=True)


def _run_kpoints_validate(data: KpointsValidateInput) -> dict[str, Any]:
    incar_request = IncarRecipeRequest.model_validate_json(
        Path(data.incar_request).read_text(encoding="utf-8")
    )
    document = read_incar(Path(data.incar))
    report = validate_kpoints(
        read_kpoints(Path(data.kpoints)).spec,
        read_poscar(Path(data.poscar)),
        incar_request.goal,
        document.spec,
    )
    report.incar_validation.diagnostics = (
        document.diagnostics + report.incar_validation.diagnostics
    )
    return report.model_dump(mode="json", exclude_none=True)


def _run_mcif_show(data: McifShowInput) -> dict[str, Any]:
    return read_mcif(Path(data.mcif), data.candidate_id).model_dump(mode="json")


def _run_workflow_from_mcif(data: WorkflowFromMcifInput) -> dict[str, Any]:
    return plan_workflow_from_mcif(
        [Path(path) for path in data.mcif],
        Path(data.output),
        workflow_id=data.workflow_id,
        task=data.task,
        magnetic_mode=data.magnetic_mode,
        encut=data.encut,
        spacing=data.spacing,
        include_dos=data.include_dos,
        dos_spacing=data.dos_spacing,
        notes=data.notes,
    )


def _run_potcar_select(data: PotcarSelectInput) -> dict[str, Any]:
    registry = load_registry(Path(data.registry))
    policy = load_policy(Path(data.policy))
    candidate = CandidateComposition.from_structure(
        read_poscar(Path(data.poscar)), data.candidate_id
    )
    goal = CalculationGoal(
        task=CalculationTask(data.task),
        magnetic_mode=MagneticMode(data.magnetic_mode),
        xc_family=registry.family,
    )
    plan = select_by_policy(candidate, goal, registry, policy)
    return plan.model_dump(mode="json")


def _prepare_workflow(data: WorkflowRequestInput):
    request = Path(data.request)
    paths = {"workflow": request, "registry": Path(data.registry), "policy": Path(data.policy)}
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    return prepare_workflow(
        Workflow.model_validate_json(payloads["workflow"]),
        PotentialRegistry.model_validate_json(payloads["registry"]),
        PotentialPolicy.model_validate_json(payloads["policy"]),
        request.resolve().parent,
        sources={
            name: {"path": str(path.resolve()), "sha256": digest(payloads[name])}
            for name, path in paths.items()
        },
    )


def _run_workflow_mae_scan(data: WorkflowMaeScanInput) -> dict[str, Any]:
    """Expand one SOC base step into a typed, local-only MAE input bundle."""

    request = Path(data.request)
    payloads = {
        "workflow": request.read_bytes(),
        "registry": Path(data.registry).read_bytes(),
        "policy": Path(data.policy).read_bytes(),
    }
    base_workflow = Workflow.model_validate_json(payloads["workflow"])
    if base_workflow.scans or len(base_workflow.steps) != 1:
        raise ValueError("MAE scan requires a workflow request containing exactly one base step")

    normalized: dict[str, list[float]] = {}
    for label, axis in data.axes.items():
        if len(axis) != 3:
            raise ValueError(f"MAE axis {label!r} must contain exactly three numbers")
        norm = sum(value * value for value in axis) ** 0.5
        if norm == 0:
            raise ValueError(f"MAE axis {label!r} must be nonzero")
        normalized[label] = [round(value / norm, 12) for value in axis]

    base = base_workflow.steps[0]
    workflow = Workflow(
        workflow_id=f"{base_workflow.workflow_id}-mae",
        steps=[],
        scans=[ParameterScan(kind="mae", base=base, axes=normalized)],
        capabilities=base_workflow.capabilities,
    )
    prepared = prepare_workflow(
        workflow,
        PotentialRegistry.model_validate_json(payloads["registry"]),
        PotentialPolicy.model_validate_json(payloads["policy"]),
        request.resolve().parent,
        sources={
            "workflow": {"path": str(request.resolve()), "sha256": digest(payloads["workflow"])},
            "registry": {"path": str(Path(data.registry).resolve()), "sha256": digest(payloads["registry"])},
            "policy": {"path": str(Path(data.policy).resolve()), "sha256": digest(payloads["policy"])},
        },
    )
    output = Path(data.output)
    _refuse_existing(output)
    manifest_path = materialize_workflow(prepared, output)
    inspection = inspect_bundle(manifest_path.parent)
    card = build_review_card(manifest_path.parent)
    return {
        **inspection,
        "review": card.model_dump(mode="json"),
        "awaiting_human": card.awaiting_human,
        "axes": normalized,
    }


def _run_workflow_validate(data: WorkflowRequestInput) -> dict[str, Any]:
    return _prepare_workflow(data).manifest


def _run_workflow_inspect(data: WorkflowInspectInput) -> dict[str, Any]:
    return inspect_bundle(Path(data.bundle))


def _run_workflow_review_card(data: WorkflowReviewCardInput) -> dict[str, Any]:
    return build_review_card(Path(data.bundle)).model_dump(mode="json")


def _run_workflow_review_history(data: WorkflowReviewHistoryInput) -> dict[str, Any]:
    records = load_review_history(
        species=data.species or None,
        task=data.task,
        magnetic_mode=data.magnetic_mode,
        reviewer=data.reviewer,
        limit=data.limit,
    )
    return {
        "reviews": [item.model_dump(mode="json") for item in records],
        "allowed_action": "advisory_only",
        "scientific_converged": False,
    }


def _run_skill_list(data: SkillListInput) -> dict[str, Any]:
    summaries = [item.model_dump(mode="json") for item in list_skills(Path(data.catalog))]
    return {"skills": summaries, "scripts_executable": False}


def _run_skill_show(data: SkillShowInput) -> dict[str, Any]:
    document = show_skill(Path(data.catalog), data.skill_id)
    return document.model_dump(mode="json")


def _run_skill_read(data: SkillReadInput) -> dict[str, Any]:
    document = read_skill_file(Path(data.catalog), data.skill_id, data.path)
    return document.model_dump(mode="json")


def _run_knowledge_search(data: KnowledgeSearchInput) -> dict[str, Any]:
    filters = RetrievalFilter(
        source_types=_enum_list(data.source_types, KnowledgeSourceType),
        tasks=_enum_list(data.tasks, CalculationTask),
        magnetic_modes=_enum_list(data.magnetic_modes, MagneticMode),
        validated_only=data.validated_only,
    )
    response = _retrieval_response(
        data.index, data.query, filters, data.limit, data.backend, data.model_dir
    )
    return response.model_dump(mode="json")


def _run_knowledge_answer(data: KnowledgeAnswerInput) -> dict[str, Any]:
    required = _enum_list(data.required_source_types, KnowledgeSourceType)
    request = AnswerRequest(
        question=data.question,
        claims=data.claims,
        limitations=data.limitations,
        backend=data.backend,
        filters=RetrievalFilter(
            source_types=required,
            tasks=_enum_list(data.tasks, CalculationTask),
            magnetic_modes=_enum_list(data.magnetic_modes, MagneticMode),
            validated_only=data.validated_only,
        ),
        required_source_types=required,
        limit=data.limit,
    )
    response = build_answer_response(
        request,
        Path(data.index),
        model_dir=Path(data.model_dir) if data.model_dir else None,
        human_review_requested=data.human_review,
    )
    return response.model_dump(mode="json")


def _run_execution_probe(data: ExecutionProbeInput) -> dict[str, Any]:
    settings = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    return probe_preview(settings, data.node)


def _run_execution_probe_live(data: ExecutionProbeLiveInput) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    settings = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    snapshot = collect_probe(settings, data.node)
    payload = snapshot.model_dump(mode="json")
    write_json_report(payload, output)
    return payload


def _validate_execution_target(profile: ClusterProfile, node: str) -> None:
    if node == profile.gateway_hostname or node not in profile.allowed_nodes:
        raise ValueError("NODE_FORBIDDEN: request must target an allowlisted compute node")


def _run_execution_request_create(
    data: ExecutionRequestCreateInput,
) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    _validate_execution_target(profile, data.node)
    request = ExecutionRequest(
        node=data.node,
        step_id=data.step_id,
        run_id=data.run_id,
        mpi_ranks=data.mpi_ranks,
        cpu_ids=data.cpu_ids,
    )
    payload = request.model_dump(mode="json")
    write_json_report(payload, output)
    return {
        **payload,
        "output_path": str(output.resolve()),
        "sha256": digest(output.read_bytes()),
    }


def _run_execution_transfer_request_create(
    data: ExecutionTransferRequestCreateInput,
) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    execution = ExecutionRequest.model_validate_json(
        Path(data.execution_request).read_bytes()
    )
    _validate_execution_target(profile, execution.node)
    request = TransferRequest(node=execution.node, run_id=execution.run_id)
    payload = request.model_dump(mode="json")
    write_json_report(payload, output)
    return {
        **payload,
        "output_path": str(output.resolve()),
        "sha256": digest(output.read_bytes()),
    }


def _run_execution_plan(data: ExecutionPlanInput) -> dict[str, Any]:
    snapshot = (
        ProbeSnapshot.model_validate_json(Path(data.snapshot).read_bytes())
        if data.snapshot
        else None
    )
    return prepare_execution(
        Path(data.bundle),
        ClusterProfile.model_validate_json(Path(data.profile).read_bytes()),
        ExecutionRequest.model_validate_json(Path(data.request).read_bytes()),
        snapshot,
    )


def _run_execution_resource_grant(
    data: ExecutionResourceGrantInput,
) -> dict[str, Any]:
    if data.confirm != RESOURCE_GRANT_CONFIRM:
        raise ValueError(
            "resource grant requires the exact phrase I_APPROVE_RESOURCE_GRANT"
        )
    approver = validate_human_approver(data.approver)
    output = Path(data.output)
    probe_output = Path(data.probe_output)
    _refuse_existing(output)
    _refuse_existing(probe_output)
    if output.resolve() == probe_output.resolve():
        raise ValueError("resource grant and probe outputs must be different paths")

    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    request = ExecutionRequest.model_validate_json(Path(data.request).read_bytes())
    _validate_execution_target(profile, request.node)
    snapshot = collect_probe(profile, request.node)
    write_json_report(snapshot.model_dump(mode="json"), probe_output)
    grant = issue_resource_grant(
        Path(data.bundle),
        profile,
        request,
        snapshot,
        output,
        approver=approver,
        confirm=data.confirm,
        now=datetime.now(UTC),
        valid_for=timedelta(minutes=data.valid_minutes),
        grant_id=data.grant_id,
    )
    return {
        **grant.model_dump(mode="json"),
        "output_path": str(output.resolve()),
        "probe_snapshot_path": str(probe_output.resolve()),
        "probe_snapshot_sha256": digest(probe_output.read_bytes()),
    }


def _collect_bound_action_probe(
    profile: ClusterProfile,
    request: ExecutionRequest,
    grant: ResourceGrant,
    output: Path,
) -> ProbeSnapshot:
    """Collect and validate a 60-second probe inside an approved action."""

    _validate_execution_target(profile, request.node)
    if not grant.matches_request(request):
        raise ValueError("RESOURCE_GRANT_MISMATCH")
    snapshot = collect_probe(profile, request.node)
    now = datetime.now(UTC)
    gate = ExecutionStateMachine(
        grant.workflow_id,
        request.step_id,
        request.run_id,
        workflow_manifest_sha256=grant.workflow_manifest_sha256,
        step_manifest_sha256=grant.step_manifest_sha256,
        run_sha256=grant.run_sha256,
        input_sha256=grant.input_sha256,
    )
    state = gate.record_resource_grant(grant, request, now=now)
    if state.stage == ExecutionStage.BLOCKED:
        raise ValueError(state.blockers[-1])
    state = gate.record_probe(snapshot, request, now=now, profile=profile)
    if state.stage == ExecutionStage.BLOCKED:
        raise ValueError(state.blockers[-1])
    write_json_report(snapshot.model_dump(mode="json"), output)
    return snapshot

def _run_execution_resource_plan(data: ExecutionResourcePlanInput) -> dict[str, Any]:
    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    request = ExecutionRequest.model_validate_json(Path(data.request).read_bytes())
    snapshot = ProbeSnapshot.model_validate_json(Path(data.snapshot).read_bytes())
    assessment = assess_resources(snapshot, request, profile=profile)
    sockets = sorted({cpu.socket for cpu in snapshot.cpus})
    physical = sorted({(cpu.socket, cpu.core) for cpu in snapshot.cpus if cpu.online})
    topo = {
        "logical_cpus": len(snapshot.cpus),
        "online_logical_cpus": sum(cpu.online for cpu in snapshot.cpus),
        "allowed_logical_cpus": len(snapshot.allowed_cpu_ids),
        "physical_cores": len(physical),
        "sockets": sockets,
        "generation_hint": "newer_or_nonuniform_topology" if len(snapshot.cpus) >= 96 or len(sockets) > 2 else "legacy_topology",
    }
    result = {"schema_version": 1, "mode": "resource_candidate", "candidate": {
        "node": request.node, "mpi_ranks": request.mpi_ranks,
        "cpu_ids": assessment.suggested_cpu_ids,
    }, "topology": topo, "assessment": assessment.model_dump(mode="json"),
    "human_review_required": True, "approved": False}
    _refuse_existing(Path(data.output))
    Path(data.output).parent.mkdir(parents=True, exist_ok=True)
    Path(data.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

def _run_execution_topss(data: ExecutionTopssSnapshotInput) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    settings = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    snapshots: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(12, len(settings.allowed_nodes))) as pool:
        futures = {
            pool.submit(collect_probe, settings, node): node
            for node in settings.allowed_nodes
        }
        for future in as_completed(futures):
            node = futures[future]
            try:
                snapshots[node] = future.result().model_dump(mode="json")
            except (ValueError, OSError) as exc:
                errors[node] = str(exc).split(":", 1)[0]
    payload = {
        "schema_version": 1,
        "mode": "topss_snapshot",
        "source": "fixed_probe_fanout",
        "observed_at": datetime.now(UTC).isoformat(),
        "read_only": True,
        "nodes": list(settings.allowed_nodes),
        "snapshots": snapshots,
        "errors": errors,
        "human_review_required": True,
        "allocation_confirmed": False,
    }
    write_json_report(payload, output)
    return payload


def _run_execution_smoke(data: ExecutionSmokeInput) -> dict[str, Any]:
    if data.confirm != SMOKE_CONFIRM:
        raise ValueError("smoke requires the exact phrase I_CONFIRM_MPI_SMOKE")
    output = Path(data.output)
    probe_output = Path(data.probe_output)
    _refuse_existing(output)
    _refuse_existing(probe_output)
    if output.resolve() == probe_output.resolve():
        raise ValueError("smoke and probe outputs must be different paths")
    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    request = ExecutionRequest.model_validate_json(Path(data.request).read_bytes())
    grant = ResourceGrant.model_validate_json(Path(data.resource_grant).read_bytes())
    _collect_bound_action_probe(profile, request, grant, probe_output)
    report = run_mpi_smoke(
        profile,
        request.node,
        confirm=data.confirm,
        run_id=request.run_id,
        ranks=grant.mpi_ranks,
        cpu_ids=grant.cpu_ids,
    )
    write_json_report(report, output)
    return report


def _run_execution_transfer(data: ExecutionTransferInput) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    bundle = Path(data.bundle).resolve()
    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    request = TransferRequest.model_validate_json(Path(data.request).read_bytes())
    grant = ResourceGrant.model_validate_json(Path(data.resource_grant).read_bytes())
    if request.node != grant.node or request.run_id != grant.run_id:
        raise ValueError("RESOURCE_GRANT_TRANSFER_MISMATCH")
    if not grant.covers(datetime.now(UTC)):
        raise ValueError("RESOURCE_GRANT_EXPIRED")
    review_path = bundle / "input.review.json"
    if review_path.is_symlink() or not review_path.is_file():
        raise ValueError("INPUT_REVIEW_NOT_ACCEPTED")
    review = InputReviewCard.model_validate_json(review_path.read_bytes())
    if (
        review.workflow_id != grant.workflow_id
        or Path(review.bundle).resolve() != bundle
        or review.status != "reviewed"
        or review.decision != "accept"
        or review.awaiting_human
        or not review.reviewer
    ):
        raise ValueError("INPUT_REVIEW_NOT_ACCEPTED")
    report = transfer_bundle(bundle, profile, request)
    write_transfer_report(report, output)
    return report


def _run_execution_remote_status(data: ExecutionRemoteStatusInput) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    record = inspect_run(Path(data.run))
    report = remote_status(profile, record)
    write_json_report(report, output)
    return report


def _run_execution_fetch(data: ExecutionFetchInput) -> dict[str, Any]:
    output = Path(data.output) if data.output is not None else None
    if output is not None:
        _refuse_existing(output)
    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    record = inspect_run(Path(data.run))
    report = fetch_results(profile, record, Path(data.destination).resolve(), data.files)
    if output is not None:
        write_json_report(report, output)
    return report


def _run_execution_artifact_release(data: ExecutionArtifactReleaseInput) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    evidence = release_chgcar(
        bundle=Path(data.bundle),
        profile=profile,
        source_run=inspect_run(Path(data.source_run)),
        source_status=Path(data.source_status),
        fetch_report=Path(data.fetch_report),
        target_transfer_report=Path(data.target_transfer_report),
        target_step=data.target_step,
        confirm=data.confirm,
    )
    payload = evidence.model_dump(mode="json")
    write_json_report(payload, output)
    return payload


def _run_potcar_build(data: PotcarBuildInput) -> dict[str, Any]:
    registry = load_registry(Path(data.registry))
    plan = PotentialPlan.model_validate_json(
        Path(data.plan).read_text(encoding="utf-8")
    )
    output = Path(data.output)
    _refuse_existing(output)
    manifest = assemble_potcar(plan, registry, output)
    payload = manifest.model_dump(mode="json")
    payload["output_path"] = str(output.resolve())
    return payload


def _run_incar_build(data: IncarBuildInput) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    request = IncarRecipeRequest.model_validate_json(
        Path(data.request).read_text(encoding="utf-8")
    )
    plan = PotentialPlan.model_validate_json(
        Path(data.plan).read_text(encoding="utf-8")
    )
    result = build_incar(
        read_poscar(Path(data.poscar)), request, plan, load_registry(Path(data.registry))
    )
    content = render_incar(result.spec)
    write_incar(result.spec, output)
    payload = result.model_dump(mode="json", exclude_none=True)
    payload["sha256"] = _sha256_text(content)
    payload["output_path"] = str(output.resolve())
    return payload


def _run_kpoints_build(data: KpointsBuildInput) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    request = KpointsRecipeRequest.model_validate_json(
        Path(data.request).read_text(encoding="utf-8")
    )
    incar_request = IncarRecipeRequest.model_validate_json(
        Path(data.incar_request).read_text(encoding="utf-8")
    )
    document = read_incar(Path(data.incar))
    result = build_kpoints(
        read_poscar(Path(data.poscar)),
        request,
        document.spec,
        incar_goal=incar_request.goal,
    )
    result.validation.incar_validation.diagnostics = (
        document.diagnostics + result.validation.incar_validation.diagnostics
    )
    content = render_kpoints(result.spec)
    write_kpoints(result.spec, output)
    payload = result.model_dump(mode="json", exclude_none=True)
    payload["sha256"] = _sha256_text(content)
    payload["output_path"] = str(output.resolve())
    return payload


def _run_workflow_build(data: WorkflowBuildInput) -> dict[str, Any]:
    output = Path(data.output)
    _refuse_existing(output)
    path = materialize_workflow(_prepare_workflow(data), output)
    inspection = inspect_bundle(path.parent)
    card = build_review_card(path.parent)
    return {
        **inspection,
        "review": card.model_dump(mode="json"),
        "awaiting_human": card.awaiting_human,
    }


def _run_execution_simulate(data: ExecutionSimulateInput) -> dict[str, Any]:
    return simulate(Path(data.output), data.scenario)


def _run_execution_inspect_run(data: ExecutionInspectRunInput) -> dict[str, Any]:
    return inspect_run(Path(data.run)).model_dump(mode="json")


def _run_execution_gate_inspect(data: ExecutionGateInspectInput) -> dict[str, Any]:
    path = Path(data.state)
    if path.is_symlink() or not path.is_file():
        raise ValueError("execution gate state must be a regular local file")
    state = ExecutionState.model_validate_json(path.read_bytes())
    next_stage = {
        ExecutionStage.VALIDATED_INPUTS: ExecutionStage.HUMAN_RESOURCE_GRANT,
        ExecutionStage.HUMAN_RESOURCE_GRANT: ExecutionStage.FRESH_PROBE,
        ExecutionStage.FRESH_PROBE: ExecutionStage.FIXED_SMOKE_TEST,
        ExecutionStage.FIXED_SMOKE_TEST: ExecutionStage.HUMAN_LAUNCH_APPROVAL,
        ExecutionStage.HUMAN_LAUNCH_APPROVAL: ExecutionStage.TRANSFER_GUARD,
        ExecutionStage.TRANSFER_GUARD: ExecutionStage.TRANSFERRED_VERIFIED,
        ExecutionStage.TRANSFERRED_VERIFIED: ExecutionStage.LAUNCH,
        ExecutionStage.LAUNCH: ExecutionStage.READONLY_STATUS,
        ExecutionStage.READONLY_STATUS: ExecutionStage.REVIEWED_RESULT,
    }.get(state.stage)
    return {
        "state": state.model_dump(mode="json"),
        "next_stage": next_stage.value if next_stage else None,
        "cluster_execution": "disabled",
        "remote_side_effect": False,
    }


def _run_knowledge_evidence(data: KnowledgeEvidenceInput) -> dict[str, Any]:
    required = _enum_list(data.required_source_types, KnowledgeSourceType)
    filters = RetrievalFilter(
        source_types=required,
        tasks=_enum_list(data.tasks, CalculationTask),
        magnetic_modes=_enum_list(data.magnetic_modes, MagneticMode),
        validated_only=data.validated_only,
    )
    response = _retrieval_response(
        data.index, data.query, filters, data.limit, data.backend, data.model_dir
    )
    package = build_evidence_package(
        response,
        backend=data.backend,
        required_source_types=required,
        human_review_requested=data.human_review,
    )
    write_evidence_package(package, Path(data.output))
    return package.model_dump(mode="json")


def _run_knowledge_draft(data: KnowledgeDraftInput) -> dict[str, Any]:
    package = EvidencePackage.model_validate_json(Path(data.evidence).read_bytes())
    draft = build_answer_draft(package, data.claims, limitations=data.limitations)
    write_answer_draft(draft, Path(data.output))
    return draft.model_dump(mode="json")


def _run_execution_transfer_plan(data: ExecutionTransferPlanInput) -> dict[str, Any]:
    settings = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    request = TransferRequest.model_validate_json(Path(data.request).read_bytes())
    report = build_transfer_plan(Path(data.bundle), settings, request)
    dumped = report.model_dump(mode="json") if isinstance(report, BaseModel) else report
    return dumped

def _run_execution_launch(data: ExecutionLaunchInput) -> dict[str, Any]:
    backend = LaunchBackend(data.backend)
    if backend != LaunchBackend.REMOTE_MPI:
        raise ValueError("execution.launch only supports the gated remote_mpi backend")
    if data.confirm != REMOTE_CONFIRM:
        raise ValueError("remote launch requires the exact phrase I_CONFIRM_REMOTE_LAUNCH")
    output = Path(data.output)
    probe_output = Path(data.probe_output)
    _refuse_existing(output)
    _refuse_existing(probe_output)
    if output.resolve() == probe_output.resolve():
        raise ValueError("launch record and probe outputs must be different paths")
    profile = ClusterProfile.model_validate_json(Path(data.profile).read_bytes())
    request = ExecutionRequest.model_validate_json(Path(data.request).read_bytes())
    grant = ResourceGrant.model_validate_json(Path(data.resource_grant).read_bytes())
    if data.run_id != request.run_id:
        raise ValueError("RUN_ID_MISMATCH")
    _collect_bound_action_probe(profile, request, grant, probe_output)
    record = launch_runtime(LaunchRequest(
        backend=backend, confirm=data.confirm, output=data.output,
        run_id=data.run_id, profile=data.profile, bundle=data.bundle,
        request=data.request, transfer_report=data.transfer_report,
        smoke_report=data.smoke_report,
        resource_grant=data.resource_grant,
        snapshot=str(probe_output),
        artifact_release=data.artifact_release,
    ))
    return record.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Registry assembly                                                             #
# --------------------------------------------------------------------------- #


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # Tier 0: read-only.
    registry.register(
        name="structure.show",
        description="Parse a VASP 5 POSCAR and return species order and atoms.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=StructureShowInput,
        run=lambda data: _run_structure_show(data),
    )
    registry.register(
        name="postprocess.parse",
        description=(
            "Parse a finished calculation directory (OSZICAR/OUTCAR/vasprun.xml) "
            "into a typed result: convergence, energy, per-ion magnetic moments, "
            "Fermi level, and cross-checks. Read-only; no cluster access."
        ),
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=PostprocessParseInput,
        run=lambda data: _run_postprocess_parse(data),
    )
    registry.register(
        name="postprocess.vaspkit",
        description="Run one fixed VASPKIT menu for band k-points, band.dat, or TDOS.dat.",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=VaspkitInput,
        run=lambda data: _run_vaspkit(data),
    )
    registry.register(
        name="postprocess.vaspkit_catalog",
        description="List the installed VASPKIT example families and integration status without executing VASPKIT.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=VaspkitCatalogInput,
        run=lambda data: _run_vaspkit_catalog(data),
    )
    registry.register(
        name="postprocess.vaspkit_task",
        description="Run a documented VASPKIT 1.5.1 numeric task with bounded menu inputs and hash every changed output.",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=VaspkitTaskInput,
        run=lambda data: _run_vaspkit_task(data),
    )
    registry.register(
        name="postprocess.plot_dos",
        description=(
            "Strictly parse a VASPKIT TDOS.dat and create a static PNG with "
            "source-data and plot SHA-256 hashes. No cluster access."
        ),
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=VaspkitPlotDosInput,
        run=lambda data: _run_vaspkit_plot_dos(data),
    )
    registry.register(
        name="postprocess.plot_band",
        description=(
            "Strictly parse VASPKIT BAND/REFORMATTED_BAND data and matching "
            "KLABELS, then create a static hashed PNG. No cluster access."
        ),
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=VaspkitPlotBandInput,
        run=lambda data: _run_vaspkit_plot_band(data),
    )
    registry.register(
        name="incar.show",
        description="Parse an INCAR into typed parameters with line diagnostics.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=IncarShowInput,
        run=lambda data: _run_incar_show(data),
    )
    registry.register(
        name="incar.validate",
        description="Cross-check an INCAR against POSCAR, plan, registry, executable.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=IncarValidateInput,
        run=lambda data: _run_incar_validate(data),
    )
    registry.register(
        name="kpoints.show",
        description="Parse a KPOINTS file into the typed sampling model.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=KpointsShowInput,
        run=lambda data: _run_kpoints_show(data),
    )
    registry.register(
        name="kpoints.validate",
        description="Check KPOINTS task consistency against POSCAR and INCAR goal.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=KpointsValidateInput,
        run=lambda data: _run_kpoints_validate(data),
    )
    registry.register(
        name="mcif.show",
        description="Parse one magnetic CIF candidate with traceable site mapping.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=McifShowInput,
        run=lambda data: _run_mcif_show(data),
    )
    registry.register(
        name="potcar.select",
        description="Create a baseline potential plan in POSCAR species order.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=PotcarSelectInput,
        run=lambda data: _run_potcar_select(data),
    )
    registry.register(
        name="workflow.validate",
        description="Compile and validate a workflow without writing any files.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=WorkflowRequestInput,
        run=lambda data: _run_workflow_validate(data),
    )
    registry.register(
        name="workflow.inspect",
        description="Verify all inputs and step-manifest hashes of a built bundle.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=WorkflowInspectInput,
        run=lambda data: _run_workflow_inspect(data),
    )
    registry.register(
        name="workflow.review_card",
        description=(
            "Show the human review card for a generated input bundle. "
            "Does not accept or reject; the user records the decision on the CLI."
        ),
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=WorkflowReviewCardInput,
        run=lambda data: _run_workflow_review_card(data),
    )
    registry.register(
        name="workflow.review_history",
        description=(
            "Read past human input-package reviews for similar species or tasks. "
            "Pass reviewer to read one member's local usage. Advisory only; "
            "never changes INCAR or launches VASP."
        ),
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=WorkflowReviewHistoryInput,
        run=lambda data: _run_workflow_review_history(data),
    )
    registry.register(
        name="skill.list",
        description=(
            "List project skills the agent may read (first-principles, literature, "
            "Reference_Demo). Scripts stay disabled."
        ),
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=SkillListInput,
        run=lambda data: _run_skill_list(data),
    )
    registry.register(
        name="skill.show",
        description="Read one skill's SKILL.md instructions. Never executes skill scripts.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=SkillShowInput,
        run=lambda data: _run_skill_show(data),
    )
    registry.register(
        name="skill.read",
        description="Read a Markdown or text reference file inside a listed skill.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=SkillReadInput,
        run=lambda data: _run_skill_read(data),
    )
    registry.register(
        name="knowledge.search",
        description="Search a local retrieval index and return traceable evidence.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=KnowledgeSearchInput,
        run=lambda data: _run_knowledge_search(data),
    )
    registry.register(
        name="knowledge.answer",
        description="Offline retrieval to a review-only [E#]-cited answer response.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=KnowledgeAnswerInput,
        run=lambda data: _run_knowledge_answer(data),
    )
    registry.register(
        name="execution.probe",
        description="Preview the fixed read-only SSH probe; never connects.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=ExecutionProbeInput,
        run=lambda data: _run_execution_probe(data),
    )
    registry.register(
        name="execution.probe_live",
        description=(
            "Run the fixed batch-key, read-only SSH probe on one allowlisted "
            "node and persist a typed ProbeSnapshot."
        ),
        risk_tier=RiskTier.CLUSTER_PLAN,
        produces_files=True,
        input_model=ExecutionProbeLiveInput,
        run=lambda data: _run_execution_probe_live(data),
    )
    registry.register(
        name="execution.plan",
        description="Offline input-integrity check and blocked execution preview.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=ExecutionPlanInput,
        run=lambda data: _run_execution_plan(data),
    )
    registry.register(
        name="execution.resource_plan",
        description="Propose node, MPI ranks, and CPU binding from a probe snapshot; requires human review and never launches.",
        risk_tier=RiskTier.CLUSTER_PLAN,
        produces_files=True,
        input_model=ExecutionResourcePlanInput,
        run=lambda data: _run_execution_resource_plan(data),
    )
    registry.register(
        name="execution.topss_snapshot",
        description=(
            "Collect fixed read-only probes from the allowlisted nodes as a "
            "resource reference; it never establishes an allocation."
        ),
        risk_tier=RiskTier.CLUSTER_PLAN,
        produces_files=True,
        input_model=ExecutionTopssSnapshotInput,
        run=lambda data: _run_execution_topss(data),
    )
    registry.register(
        name="execution.inspect_run",
        description=(
            "Read a local launch/status record. Does not contact the cluster, "
            "start MPI, or cancel any process."
        ),
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=ExecutionInspectRunInput,
        run=lambda data: _run_execution_inspect_run(data),
    )
    registry.register(
        name="execution.remote_status",
        description=(
            "Query only the recorded remote PID and allowlisted VASP status "
            "files for one controlled run; never signals a process."
        ),
        risk_tier=RiskTier.CLUSTER_PLAN,
        produces_files=True,
        input_model=ExecutionRemoteStatusInput,
        run=lambda data: _run_execution_remote_status(data),
    )
    registry.register(
        name="execution.fetch",
        description=(
            "Fetch exact allowlisted result basenames for a controlled run into "
            "a new local directory and verify remote/local SHA-256 hashes."
        ),
        risk_tier=RiskTier.CLUSTER_PLAN,
        produces_files=True,
        input_model=ExecutionFetchInput,
        run=lambda data: _run_execution_fetch(data),
    )
    registry.register(
        name="execution.artifact_release",
        description=(
            "Promote a verified, converged SCF CHGCAR to one declared dependent "
            "step using a fixed remote protocol; requires separate named approval."
        ),
        risk_tier=RiskTier.CLUSTER_EXECUTE,
        produces_files=True,
        input_model=ExecutionArtifactReleaseInput,
        run=lambda data: _run_execution_artifact_release(data),
    )
    registry.register(
        name="execution.gate_inspect",
        description=(
            "Inspect a local execution-gate state and report the next required "
            "approval; never connects, uploads, launches, cancels, or retries."
        ),
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=ExecutionGateInspectInput,
        run=lambda data: _run_execution_gate_inspect(data),
    )

    # Tier 1: local writes / dry-run generation.
    registry.register(
        name="potcar.build",
        description="Assemble a POTCAR after validating names, order, and hashes.",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=PotcarBuildInput,
        run=lambda data: _run_potcar_build(data),
    )
    registry.register(
        name="incar.build",
        description="Generate an INCAR from a typed request; refuses overwrite.",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=IncarBuildInput,
        run=lambda data: _run_incar_build(data),
    )
    registry.register(
        name="kpoints.build",
        description="Generate task-matched KPOINTS; refuses overwrite.",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=KpointsBuildInput,
        run=lambda data: _run_kpoints_build(data),
    )
    registry.register(
        name="workflow.from_mcif",
        description=(
            "Create a typed workflow request from one or more mCIF files. "
            "MAGMOM is mapped from the candidate; the model must not invent moments. "
            "Writes the request JSON only."
        ),
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=WorkflowFromMcifInput,
        run=lambda data: _run_workflow_from_mcif(data),
    )
    registry.register(
        name="workflow.build",
        description="Materialize a local input bundle; never runs VASP.",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=WorkflowBuildInput,
        run=lambda data: _run_workflow_build(data),
    )
    registry.register(
        name="workflow.mae_scan",
        description=(
            "Materialize independent SOC input directories for each requested "
            "magnetization axis; changes only SAXIS and never submits VASP."
        ),
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=WorkflowMaeScanInput,
        run=lambda data: _run_workflow_mae_scan(data),
    )
    registry.register(
        name="execution.simulate",
        description="Run a fixed tiny local process, not VASP, SSH, or MPI.",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=ExecutionSimulateInput,
        run=lambda data: _run_execution_simulate(data),
    )
    registry.register(
        name="execution.request_create",
        description=(
            "Write one allowlisted, typed ExecutionRequest. This creates local "
            "intent only and does not allocate resources or contact the cluster."
        ),
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=ExecutionRequestCreateInput,
        run=lambda data: _run_execution_request_create(data),
    )
    registry.register(
        name="execution.transfer_request_create",
        description=(
            "Write one allowlisted, no-overwrite TransferRequest for the same "
            "run identity; never connects or uploads."
        ),
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=ExecutionTransferRequestCreateInput,
        run=lambda data: _run_execution_transfer_request_create(data),
    )
    registry.register(
        name="knowledge.evidence",
        description="Create an immutable, cited evidence package (draft_only).",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=KnowledgeEvidenceInput,
        run=lambda data: _run_knowledge_evidence(data),
    )
    registry.register(
        name="knowledge.draft",
        description="Create a human-reviewable [E#]-cited draft (draft_only).",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=KnowledgeDraftInput,
        run=lambda data: _run_knowledge_draft(data),
    )

    # Elevated operations. Every call pauses for a named human approval.
    registry.register(
        name="execution.transfer_plan",
        description="Build an offline SFTP transfer plan; never connects.",
        risk_tier=RiskTier.CLUSTER_PLAN,
        produces_files=False,
        input_model=ExecutionTransferPlanInput,
        run=lambda data: _run_execution_transfer_plan(data),
    )
    registry.register(
        name="execution.resource_grant",
        description=(
            "After named cluster_plan approval, collect a fresh fixed probe and "
            "issue an accepted-input ResourceGrant only with the exact approval phrase."
        ),
        risk_tier=RiskTier.CLUSTER_PLAN,
        produces_files=True,
        input_model=ExecutionResourceGrantInput,
        run=lambda data: _run_execution_resource_grant(data),
    )
    registry.register(
        name="execution.smoke",
        description=(
            "Collect an action-local fresh probe, then run the fixed hostname/CPU "
            "Intel MPI smoke bound to one current ResourceGrant; never starts VASP."
        ),
        risk_tier=RiskTier.CLUSTER_EXECUTE,
        produces_files=True,
        input_model=ExecutionSmokeInput,
        run=lambda data: _run_execution_smoke(data),
    )
    registry.register(
        name="execution.transfer",
        description=(
            "Upload one accepted immutable input bundle to a new run directory "
            "using fixed SFTP and verify every remote hash."
        ),
        risk_tier=RiskTier.CLUSTER_EXECUTE,
        produces_files=True,
        input_model=ExecutionTransferInput,
        run=lambda data: _run_execution_transfer(data),
    )
    registry.register(
        name="execution.launch",
        description=(
            "After a separate named approval, collect a fresh launch-time probe "
            "and start fixed remote VASP only when grant, smoke, and transfer bind."
        ),
        risk_tier=RiskTier.CLUSTER_EXECUTE,
        produces_files=True,
        input_model=ExecutionLaunchInput,
        run=lambda data: _run_execution_launch(data),
    )

    return registry
