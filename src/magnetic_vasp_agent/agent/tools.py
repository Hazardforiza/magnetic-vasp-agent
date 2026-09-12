"""Typed tool registry exposing the deterministic VASP core to an agent.

Each tool wraps an existing deterministic function. Tools never execute
arbitrary shell, never connect to the cluster, and never place POTCAR content in
their result. Read-only tools (``read_only``) only parse or validate; local
writes (``local_write``) generate files under an output path chosen by the
caller; ``cluster_plan`` tools only produce an offline transfer plan (no
``--connect``). No ``cluster_execute`` tool is ever registered.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from magnetic_vasp_agent.agent.tool_models import (
    RiskTier,
    StrictModel,
    ToolSpec,
)
from magnetic_vasp_agent.execution import prepare_execution
from magnetic_vasp_agent.execution_models import (
    ClusterProfile,
    ExecutionRequest,
    ProbeSnapshot,
)
from magnetic_vasp_agent.execution_probe import probe_preview
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
from magnetic_vasp_agent.potcar import (
    assemble_potcar,
    load_policy,
    load_registry,
    select_by_policy,
)
from magnetic_vasp_agent.transfer import build_transfer_plan
from magnetic_vasp_agent.transfer_models import TransferRequest
from magnetic_vasp_agent.workflow import digest, inspect_bundle, materialize_workflow, prepare_workflow
from magnetic_vasp_agent.workflow_models import Workflow


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
        if risk_tier == RiskTier.CLUSTER_EXECUTE:
            raise ValueError("cluster_execute tools must never be registered")
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


class WorkflowInspectInput(StrictModel):
    bundle: str = Field(min_length=1)


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


class ExecutionPlanInput(StrictModel):
    bundle: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    request: str = Field(min_length=1)
    snapshot: str | None = None


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


class PostprocessParseInput(StrictModel):
    directory: str = Field(min_length=1)


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


def _run_workflow_validate(data: WorkflowRequestInput) -> dict[str, Any]:
    return _prepare_workflow(data).manifest


def _run_workflow_inspect(data: WorkflowInspectInput) -> dict[str, Any]:
    return inspect_bundle(Path(data.bundle))


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
    return inspect_bundle(path.parent)


def _run_execution_simulate(data: ExecutionSimulateInput) -> dict[str, Any]:
    return simulate(Path(data.output), data.scenario)


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
        name="execution.plan",
        description="Offline input-integrity check and blocked execution preview.",
        risk_tier=RiskTier.READ_ONLY,
        produces_files=False,
        input_model=ExecutionPlanInput,
        run=lambda data: _run_execution_plan(data),
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
        name="workflow.build",
        description="Materialize a local input bundle; never runs VASP.",
        risk_tier=RiskTier.LOCAL_WRITE,
        produces_files=True,
        input_model=WorkflowBuildInput,
        run=lambda data: _run_workflow_build(data),
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

    # Tier 2: cluster plan (offline plan only, no --connect).
    registry.register(
        name="execution.transfer_plan",
        description="Build an offline SFTP transfer plan; never connects.",
        risk_tier=RiskTier.CLUSTER_PLAN,
        produces_files=False,
        input_model=ExecutionTransferPlanInput,
        run=lambda data: _run_execution_transfer_plan(data),
    )

    return registry
