"""Local browser interface for reviewed VASP workflows and results."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from magnetic_vasp_agent.execution_models import ClusterProfile, ExecutionRequest
from magnetic_vasp_agent.execution_probe import collect_probe, collect_wannier_probe
from magnetic_vasp_agent.input_review import build_review_card, record_input_review
from magnetic_vasp_agent.input_review_models import REVIEWER_BLOCKLIST
from magnetic_vasp_agent.knowledge.library import KnowledgeLibrary
from magnetic_vasp_agent.mcif_workflow import plan_workflow_from_mcif
from magnetic_vasp_agent.models import PotentialPolicy, PotentialRegistry
from magnetic_vasp_agent.postprocess import analyze_calculation
from magnetic_vasp_agent.project_resources import (
    ResourcePreferences,
    ResourceSelectionError,
    choose_project_resources,
    validate_preferences,
)
from magnetic_vasp_agent.research import ResearchConfig, plan_research_from_mcif
from magnetic_vasp_agent.vaspkit_plot import plot_band, plot_tdos
from magnetic_vasp_agent.web_knowledge import create_knowledge_router
from magnetic_vasp_agent.workflow import (
    digest,
    inspect_bundle,
    materialize_workflow,
    prepare_workflow,
)
from magnetic_vasp_agent.workflow_models import Workflow

_ID = re.compile(r"[0-9a-f]{32}\Z")
_REVIEWER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_RESULT_FILES = {
    "OSZICAR", "OUTCAR", "vasprun.xml", "POSCAR", "TDOS.dat",
    "BAND.dat", "REFORMATTED_BAND.dat", "KLABELS",
}
_PLOT_FILES = {"dos": "dos.png", "band": "band.png"}
_MCIF_LIMIT = 4 * 1024 * 1024
_RESULT_FILE_LIMIT = 128 * 1024 * 1024
_RESULT_TOTAL_LIMIT = 256 * 1024 * 1024
_MAX_SESSIONS = 100
_PLOT_PROVENANCE = "unverified_external_tables"
_NODE_OBSERVATION_NOTE = "节点状态是只读瞬时观察，不代表资源已分配或获得运行授权。"
_PROJECT_RESEARCH_TASKS = {"magnetic_order", "mae", "convergence"}


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer: str = Field(min_length=1, max_length=64)
    decision: Literal["accept", "reject", "revise"]
    notes: str | None = Field(default=None, max_length=4000)


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    goal: str = Field(min_length=1, max_length=4000)
    project_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class _ProjectResources(ResourcePreferences):
    revision: int = Field(default=0, ge=0)
    selection: dict[str, Any] | None = None


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer: str = Field(min_length=1, max_length=64)
    decisions: dict[str, bool] = Field(min_length=1)


@dataclass
class _AgentSession:
    id: str
    goal: str
    status: str = "running"
    graph: Any = None
    result: Any = None
    error: str | None = None
    project_id: str | None = None
    resource_revision: int | None = None
    resource_request: ExecutionRequest | None = None
    request_path: Path | None = None


def _human_reviewer(value: str) -> str:
    name = value.strip()
    if _REVIEWER.fullmatch(name) is None or name.casefold() in REVIEWER_BLOCKLIST:
        raise HTTPException(422, "reviewer must identify a named human")
    return name


def _new_id() -> str:
    return uuid.uuid4().hex


def _owned_directory(base: Path, item_id: str) -> Path:
    if _ID.fullmatch(item_id) is None:
        raise HTTPException(404, "item not found")
    target = base / item_id
    if target.is_symlink() or not target.is_dir() or target.resolve().parent != base.resolve():
        raise HTTPException(404, "item not found")
    return target


def _owned_bundle(folder: Path) -> Path:
    bundle = folder / "bundle"
    if bundle.is_symlink() or not bundle.is_dir() or bundle.resolve().parent != folder.resolve():
        raise HTTPException(404, "project bundle not found")
    return bundle


def _discard_new_directory(base: Path, item_id: str) -> None:
    """Remove only the UUID directory this request just allocated."""

    target = base / item_id
    if _ID.fullmatch(item_id) and not target.is_symlink() and target.resolve().parent == base.resolve():
        shutil.rmtree(target)


async def _save_upload(upload: UploadFile, destination: Path, limit: int) -> int:
    total = 0
    with destination.open("xb") as handle:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                raise HTTPException(413, "uploaded file exceeds size limit")
            handle.write(chunk)
    if total == 0:
        raise HTTPException(422, "uploaded file is empty")
    return total


def create_app(
    *,
    root: Path | None = None,
    registry: Path | None = None,
    policy: Path | None = None,
    llm_config: Path | None = None,
    execution_profile: Path | None = None,
) -> FastAPI:
    """Create a loopback-only app; its paths come from the server operator."""

    data_root = Path(root or "var/web").resolve()
    registry_path = Path(registry or "var/potcar_registry.json").resolve()
    policy_path = Path(policy or "config/potential_policy.json").resolve()
    llm_path = Path(llm_config or "config/agent/llm.json").resolve()
    execution_profile_path = Path(
        execution_profile or "config/execution/cluster.json"
    ).resolve()
    projects_root = data_root / "projects"
    results_root = data_root / "results"
    review_log = data_root / "reviews" / "input-reviews.jsonl"
    projects_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    csrf_token = secrets.token_urlsafe(32)
    sessions: dict[str, _AgentSession] = {}
    session_lock = threading.RLock()
    review_lock = threading.RLock()
    resource_lock = threading.RLock()
    selecting_projects: set[str] = set()
    knowledge_library = KnowledgeLibrary(data_root)
    app = FastAPI(title="MagVASP Local Web", docs_url=None, redoc_url=None, openapi_url=None)

    def load_execution_profile() -> tuple[ClusterProfile | None, str | None]:
        """Load the private profile for read-only node observation.

        The profile is deliberately read per request so a local operator can
        regenerate it without restarting the web service.  Only its allowlist
        and public runtime markers are ever sent to the browser.
        """

        if not execution_profile_path.is_file():
            return None, "cluster profile is not configured"
        try:
            return (
                ClusterProfile.model_validate_json(execution_profile_path.read_bytes()),
                None,
            )
        except (ValueError, OSError, TypeError) as exc:
            return None, f"cluster profile is invalid: {type(exc).__name__}"

    def public_probe_error(exc: Exception) -> dict[str, str]:
        """Return a stable, non-sensitive error without remote stderr."""

        text = str(exc)
        known = {
            "SSH_PROBE_FAILED": "无法连接或验证计算节点",
            "SSH_WANNIER_PROBE_FAILED": "无法检查 Wannier 运行环境",
            "PROBE_TOO_LARGE": "节点探测返回内容过大",
            "WANNIER_PROBE_TOO_LARGE": "Wannier 探测返回内容过大",
            "NODE_FORBIDDEN": "节点不在配置的白名单中",
        }
        code = text.split(":", 1)[0] if text else "PROBE_FAILED"
        if code not in known:
            code = "PROBE_INVALID" if "probe" in text.lower() else "PROBE_FAILED"
        return {"code": code, "message": known.get(code, "节点探测失败")}

    def summarise_node_probe(profile: ClusterProfile, node: str) -> dict[str, Any]:
        """Collect and redact the fixed node and Wannier observations."""

        try:
            snapshot = collect_probe(profile, node, interactive=False)
        except (ValueError, OSError) as exc:
            return {
                "name": node,
                "status": "connection_failed",
                "status_reason": "节点探测失败",
                "error": public_probe_error(exc),
                "wannier": {"status": "unknown"},
                "observation_note": _NODE_OBSERVATION_NOTE,
            }

        online_cpus = [cpu for cpu in snapshot.cpus if cpu.online]
        accessible_cpus = [cpu for cpu in online_cpus if cpu.cpu in snapshot.allowed_cpu_ids]
        total_cpu_percent = round(sum(item.cpu_percent for item in snapshot.processes), 2)
        busy = snapshot.load1 >= 0.5 or total_cpu_percent >= 5.0
        status = "busy" if busy else "observable"
        status_reason = "观察到 CPU 活动或负载" if busy else "当前探测未观察到明显 CPU 活动"
        payload = {
            "name": node,
            "status": status,
            "status_reason": status_reason,
            "observed_at": snapshot.observed_at.isoformat(),
            "snapshot": {
                "cpu_count": len(online_cpus),
                "physical_core_count": len({(cpu.socket, cpu.core) for cpu in online_cpus}),
                "accessible_physical_core_count": len({(cpu.socket, cpu.core) for cpu in accessible_cpus}),
                "accessible_cpu_count": len(
                    set(snapshot.allowed_cpu_ids).intersection(cpu.cpu for cpu in online_cpus)
                ),
                "load1": snapshot.load1,
                "mem_available_kib": snapshot.mem_available_kib,
                "mem_available_gib": round(snapshot.mem_available_kib / (1024 * 1024), 2),
                "process_count": len(snapshot.processes),
                "active_process_count": sum(
                    item.cpu_percent >= 5 for item in snapshot.processes
                ),
                "total_cpu_percent": total_cpu_percent,
                "vasp": {
                    "module_loaded": snapshot.module_loaded,
                    "mpi_path_observed": snapshot.mpi_path == profile.mpi_path,
                    "executables": snapshot.executables,
                },
            },
            "wannier": {"status": "checking"},
            "observation_note": _NODE_OBSERVATION_NOTE,
        }
        try:
            wannier = collect_wannier_probe(profile, node, interactive=False)
        except (ValueError, OSError) as exc:
            payload["wannier"] = {
                "status": "connection_failed",
                "error": public_probe_error(exc),
            }
        else:
            payload["wannier"] = {
                "status": "verified" if wannier.capability_verified else "incomplete",
                "supported": wannier.capability_verified,
                "module_loaded": wannier.module_loaded,
                "version": wannier.wannier90_version,
                "wannier90_path": wannier.wannier90_path,
                "vasp_ncl_path": wannier.vasp_ncl_path,
                "observed_at": wannier.observed_at.isoformat(),
            }
        return payload

    @app.middleware("http")
    async def local_guard(request: Request, call_next):
        try:
            host = urlsplit("http://" + request.headers.get("host", "")).hostname
        except ValueError:
            host = None
        if host not in _LOCAL_HOSTS:
            return JSONResponse({"detail": "web interface is available only on localhost"}, status_code=403)
        origin = request.headers.get("origin")
        if origin is not None:
            try:
                parsed = urlsplit(origin)
            except ValueError:
                return JSONResponse({"detail": "cross-origin requests are not allowed"}, status_code=403)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != request.headers.get("host"):
                return JSONResponse({"detail": "cross-origin requests are not allowed"}, status_code=403)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            if not secrets.compare_digest(request.headers.get("x-csrf-token", ""), csrf_token):
                return JSONResponse({"detail": "missing or invalid CSRF token"}, status_code=403)
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    too_large = int(length) > _RESULT_TOTAL_LIMIT + 1024 * 1024
                except ValueError:
                    return JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
                if too_large:
                    return JSONResponse({"detail": "request exceeds size limit"}, status_code=413)
        return await call_next(request)

    def read_resources(folder: Path) -> _ProjectResources:
        source = folder / "resources.json"
        if source.is_symlink():
            raise HTTPException(422, "project resource settings must not be a link")
        if not source.exists():
            return _ProjectResources()
        try:
            if source.stat().st_size > 1024 * 1024:
                raise ValueError("resource settings are too large")
            return _ProjectResources.model_validate_json(source.read_bytes())
        except (ValueError, OSError) as exc:
            raise HTTPException(422, "project resource settings are invalid") from exc

    def write_resources(folder: Path, resources: _ProjectResources) -> None:
        # Keep resource choices outside the hashed scientific bundle. Replacing
        # one local file also keeps concurrent reads from seeing partial JSON.
        target = folder / "resources.json"
        if target.is_symlink():
            raise HTTPException(422, "project resource settings must not be a link")
        temporary = folder / f".resources-{_new_id()}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(resources.model_dump_json(indent=2))
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def checked_preferences(preferences: ResourcePreferences) -> ResourcePreferences:
        profile, _ = load_execution_profile()
        try:
            return validate_preferences(preferences, profile)
        except ResourceSelectionError as exc:
            raise HTTPException(422, str(exc)) from exc

    def profile_fingerprint(profile: ClusterProfile | None) -> str | None:
        return digest(profile.model_dump_json().encode("utf-8")) if profile is not None else None

    def project_detail(item_id: str) -> dict[str, Any]:
        folder = _owned_directory(projects_root, item_id)
        bundle = _owned_bundle(folder)
        card = build_review_card(bundle, log_path=review_log)
        steps = []
        for step in card.steps:
            step_dir = bundle / step.step_id
            if step_dir.is_symlink() or not step_dir.is_dir() or step_dir.resolve().parent != bundle.resolve():
                raise HTTPException(422, "step directory is outside its project")
            inputs = {}
            for filename in ("POSCAR", "INCAR", "KPOINTS"):
                source = step_dir / filename
                if source.is_symlink() or source.resolve().parent != step_dir.resolve() or source.stat().st_size > 1024 * 1024:
                    raise HTTPException(422, "input file is too large or is a link")
                inputs[filename] = source.read_text(encoding="utf-8")
            steps.append(
                {
                    "step_id": step.step_id,
                    "inputs": inputs,
                    "potcar": {
                        "sha256": step.file_hashes.get("POTCAR"),
                        "potential_names": step.potcar_names,
                    },
                }
            )
        return {"id": item_id, "review_card": card.model_dump(mode="json"), "steps": steps,
                "resources": read_resources(folder).model_dump(mode="json")}

    def result_detail(item_id: str) -> dict[str, Any]:
        folder = _owned_directory(results_root, item_id)
        summary_file = folder / "summary.json"
        if summary_file.is_symlink() or not summary_file.is_file():
            raise HTTPException(404, "result not found")
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        plots = {
            kind: f"/api/results/{item_id}/plots/{kind}"
            for kind, filename in _PLOT_FILES.items()
            if (folder / filename).is_file() and not (folder / filename).is_symlink()
        }
        return {
            "id": item_id,
            "summary": summary,
            "plots": plots,
            "plot_provenance": _PLOT_PROVENANCE if plots else None,
        }

    def knowledge_project_summary(item_id: str) -> dict[str, Any]:
        # Use only the deterministic review card; do not copy input bodies,
        # private cluster profiles, local filesystem roots or resource requests.
        detail = project_detail(item_id)
        card = detail["review_card"]
        return {
            "project_id": item_id,
            "workflow_id": card.get("workflow_id"),
            "input_review_status": card.get("status"),
            "input_review_decision": card.get("decision"),
            "integrity_valid": card.get("integrity_valid"),
            "steps": [{key: step.get(key) for key in (
                "step_id", "task", "magnetic_mode", "executable", "species_order",
                "counts", "encut", "magmom", "potcar_names", "kpoints_summary", "file_hashes",
            )} for step in card.get("steps", [])],
            "scientific_validation": "unverified",
            "note": "输入审核只确认输入方案，不代表计算完成、收敛或科学验证。",
        }

    def knowledge_result_summary(item_id: str) -> dict[str, Any]:
        detail = result_detail(item_id)
        # Result models contain a local directory and diagnostic paths. Select
        # scientific summary fields explicitly before publishing a case.
        summary = detail["summary"]
        allowed = (
            "converged", "ionic_converged", "electronic_converged", "final_energy",
            "energy_per_atom", "fermi_energy", "total_magnetization", "magnetic_mode",
            "total_magnetization_vector", "max_force", "result_sha256",
            "n_ionic_steps", "n_electronic_steps", "atom_count", "vasp_version",
        )
        source_evidence = {}
        for source in ("oszicar", "outcar", "vasprun"):
            parsed = summary.get(source)
            if isinstance(parsed, dict):
                source_evidence[source] = {key: parsed[key] for key in (
                    "source_sha256", "vasp_version", "electronic_converged",
                    "ionic_converged", "ionic_steps", "electronic_steps_last",
                ) if key in parsed}
        return {
            "result_id": item_id,
            "summary": {key: summary[key] for key in allowed if key in summary},
            "source_evidence": source_evidence,
            "warning_count": len(summary.get("warnings", [])),
            "scientific_validation": "unverified",
            "note": "导入结果的自动分析不是人工科学验证，项目关联由用户提供。",
        }

    app.include_router(create_knowledge_router(
        knowledge_library, project_summary=knowledge_project_summary,
        result_summary=knowledge_result_summary,
    ))

    def session_detail(item_id: str) -> dict[str, Any]:
        with session_lock:
            session = sessions.get(item_id)
            if session is None:
                raise HTTPException(404, "agent session not found")
            result = session.result
            return {
                "id": item_id,
                "status": session.status,
                "result": result.model_dump(mode="json") if result is not None else None,
                "pending_approvals": [
                    item.model_dump(mode="json") for item in result.pending_approvals
                ] if result is not None else [],
                "error": session.error,
                "project_id": session.project_id,
                "resource_revision": session.resource_revision,
            }

    def finish_session(item_id: str, result: Any, graph: Any) -> None:
        with session_lock:
            session = sessions[item_id]
            session.graph = graph
            session.result = result
            session.status = "pending_approval" if result.pending_approvals else "completed"

    def fail_session(item_id: str, exc: Exception) -> None:
        with session_lock:
            session = sessions[item_id]
            session.status = "error"
            session.error = str(exc)

    def run_session(item_id: str) -> None:
        try:
            from magnetic_vasp_agent.agent.graph import VaspAgentGraph
            from magnetic_vasp_agent.agent.library_context import register_library_search
            from magnetic_vasp_agent.agent.llm_config import load_llm_config

            session = sessions[item_id]
            if session.project_id is not None:
                from magnetic_vasp_agent.agent.project_context import project_registry

                folder = _owned_directory(projects_root, session.project_id)

                def validate_context() -> None:
                    current = read_resources(folder)
                    if current.revision != session.resource_revision or current.selection is None:
                        raise ValueError("PROJECT_RESOURCES_CHANGED: create a new plan after changing resources")
                    if current.selection.get("profile_sha256") != profile_fingerprint(load_execution_profile()[0]):
                        raise ValueError("PROJECT_PROFILE_CHANGED: check resources again with the current cluster profile")
                    stored = ExecutionRequest.model_validate(current.selection["execution_request"])
                    if stored != session.resource_request:
                        raise ValueError("PROJECT_RESOURCES_CHANGED: selected execution request changed")

                validate_context()
                registry = project_registry(
                    execution_profile_path, _owned_bundle(folder), session.resource_request,
                    request_path=session.request_path, validate_context=validate_context,
                )
            else:
                from magnetic_vasp_agent.agent.tools import default_registry

                registry = default_registry()
            register_library_search(registry, knowledge_library.search)
            graph = VaspAgentGraph(load_llm_config(llm_path), registry=registry)
            result = graph.run(session.goal)
            finish_session(item_id, result, graph)
        except Exception as exc:
            fail_session(item_id, exc)

    def resume_session(item_id: str, decisions: dict[str, dict[str, Any]]) -> None:
        try:
            with session_lock:
                session = sessions[item_id]
                graph, previous = session.graph, session.result
            result = graph.resume(previous, decisions)
            finish_session(item_id, result, graph)
        except Exception as exc:
            fail_session(item_id, exc)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        agent_ready = False
        if llm_path.is_file():
            try:
                from magnetic_vasp_agent.agent.graph import langgraph_available
                from magnetic_vasp_agent.agent.llm_config import load_llm_config

                agent_ready = langgraph_available() and bool(load_llm_config(llm_path).resolve_api_key())
            except (ValueError, RuntimeError, OSError):
                pass
        return {
            "ready": True,
            "csrf_token": csrf_token,
            "agent_configured": agent_ready,
            "project_generation_ready": registry_path.is_file() and policy_path.is_file(),
            "execution_profile_configured": load_execution_profile()[0] is not None,
        }

    @app.get("/api/nodes")
    def list_nodes() -> dict[str, Any]:
        """List the configured compute allowlist without contacting the cluster."""

        profile, error = load_execution_profile()
        if profile is None:
            return {
                "configured": False,
                "nodes": [],
                "error": error,
                "observation_note": _NODE_OBSERVATION_NOTE,
            }
        return {
            "configured": True,
            "nodes": [{"name": node, "status": "unknown"} for node in profile.allowed_nodes
                      if node.casefold() != "mu01"],
            "observation_note": _NODE_OBSERVATION_NOTE,
        }

    @app.get("/api/nodes/{node}/probe")
    def probe_node(node: str) -> dict[str, Any]:
        """Run fixed read-only probes for one allowlisted compute node."""

        profile, error = load_execution_profile()
        if profile is None:
            raise HTTPException(503, error or "cluster profile is not configured")
        if node.casefold() == "mu01" or node not in profile.allowed_nodes:
            raise HTTPException(404, "node is not in the configured cluster profile")
        return summarise_node_probe(profile, node)

    @app.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        items = []
        for folder in sorted(projects_root.iterdir(), key=lambda path: path.name, reverse=True):
            if not _ID.fullmatch(folder.name) or folder.is_symlink() or not folder.is_dir():
                continue
            try:
                bundle = _owned_bundle(folder)
            except HTTPException:
                continue
            card = build_review_card(bundle, log_path=review_log)
            items.append({"id": folder.name, "workflow_id": card.workflow_id,
                          "status": card.status, "decision": card.decision,
                          "step_count": len(card.steps)})
        return {"projects": items}

    @app.post("/api/projects", status_code=201)
    async def create_project(
        file: UploadFile = File(...),
        task: str = Form("static"),
        encut: float = Form(600.0, ge=100, le=2000),
        spacing: float = Form(0.25, gt=0, le=1),
        include_dos: bool = Form(False),
        resource_mode: Literal["manual", "llm"] = Form("llm"),
        resource_node: str | None = Form(None),
        resource_mpi_ranks: int | None = Form(None, ge=1, le=1024),
    ) -> dict[str, Any]:
        if Path(file.filename or "").suffix.lower() not in {".mcif", ".cif"}:
            raise HTTPException(422, "upload an .mcif or .cif file")
        if not registry_path.is_file() or not policy_path.is_file():
            raise HTTPException(503, "POTCAR registry or potential policy is not configured")
        try:
            preferences = checked_preferences(ResourcePreferences(
                mode=resource_mode, node=resource_node, mpi_ranks=resource_mpi_ranks,
            ))
        except ValueError as exc:
            raise HTTPException(422, "节点与核数设置无效，请检查选择方式和必填项。") from exc
        item_id = _new_id()
        folder = projects_root / item_id
        folder.mkdir(mode=0o700)
        source = folder / "source.mcif"
        complete = False
        try:
            await _save_upload(file, source, _MCIF_LIMIT)
            if task not in {"static", "relax", *_PROJECT_RESEARCH_TASKS}:
                raise ValueError(f"unsupported project task: {task}")
            if include_dos and task != "static":
                raise ValueError("DOS follow-up is only available for a static SCF task")
            advanced = task in _PROJECT_RESEARCH_TASKS
            if advanced:
                research_dir = folder / "research"
                plan = plan_research_from_mcif(
                    [source], research_dir, plan_id=item_id,
                    config=ResearchConfig(
                        tasks=[task],
                        encut=encut,
                        kpoint_spacing=spacing,
                    ),
                )
                if plan.workflow_request is None:
                    raise ValueError(
                        "selected research tasks did not produce executable VASP steps"
                    )
                mae_stage = next((stage for stage in plan.stages if stage.task == "mae"), None)
                if task == "mae" and (mae_stage is None or mae_stage.status != "workflow_generated"):
                    raise ValueError("MAE requires a collinear mCIF with explicit SOC directions")
                request_file = research_dir / plan.workflow_request
                workflow_base = research_dir
            else:
                request_file = folder / "workflow.request.json"
                plan_workflow_from_mcif(
                    [source], request_file, workflow_id=item_id,
                    task=task, encut=encut, spacing=spacing,
                    include_dos=include_dos,
                    review_log=review_log,
                )
                workflow_base = request_file.parent
            payloads = {
                "workflow": request_file.read_bytes(),
                "registry": registry_path.read_bytes(),
                "policy": policy_path.read_bytes(),
            }
            prepared = prepare_workflow(
                Workflow.model_validate_json(payloads["workflow"]),
                PotentialRegistry.model_validate_json(payloads["registry"]),
                PotentialPolicy.model_validate_json(payloads["policy"]),
                workflow_base,
                sources={name: {"sha256": digest(payload)} for name, payload in payloads.items()},
            )
            materialize_workflow(prepared, folder / "bundle")
            write_resources(folder, _ProjectResources(**preferences.model_dump(), revision=1))
            detail = project_detail(item_id)
            complete = True
            return detail
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            await file.close()
            if not complete:
                _discard_new_directory(projects_root, item_id)

    @app.get("/api/projects/{item_id}")
    def get_project(item_id: str) -> dict[str, Any]:
        return project_detail(item_id)

    @app.put("/api/projects/{item_id}/resources")
    def update_resources(item_id: str, body: ResourcePreferences) -> dict[str, Any]:
        folder = _owned_directory(projects_root, item_id)
        preferences = checked_preferences(body)
        with resource_lock:
            if item_id in selecting_projects:
                raise HTTPException(409, "正在检查该项目的资源，请等待完成后再修改。")
            previous = read_resources(folder)
            write_resources(folder, _ProjectResources(
                **preferences.model_dump(), revision=previous.revision + 1,
            ))
        return project_detail(item_id)

    @app.post("/api/projects/{item_id}/resources/select")
    def select_project_resources(item_id: str) -> dict[str, Any]:
        folder = _owned_directory(projects_root, item_id)
        profile, error = load_execution_profile()
        if profile is None:
            raise HTTPException(503, error or "cluster profile is not configured")
        with resource_lock:
            if item_id in selecting_projects:
                raise HTTPException(409, "正在检查该项目的资源，请等待完成。")
            previous = read_resources(folder)
            selecting_projects.add(item_id)
        try:
            card = build_review_card(_owned_bundle(folder), log_path=review_log)
            if not card.integrity_valid or not card.steps:
                raise HTTPException(422, "输入完整性校验失败，无法生成资源方案。")
            step = card.steps[0]
            preferences = ResourcePreferences(
                mode=previous.mode, node=previous.node, mpi_ranks=previous.mpi_ranks,
            )
            selection = choose_project_resources(profile, preferences, {
                "step_id": step.step_id, "run_id": "web-" + _new_id()[:24],
                "workflow_id": card.workflow_id, "step_count": len(card.steps),
                "task": step.task, "executable": step.executable,
                "atom_count": sum(step.counts), "species": step.species_order,
                "encut": step.encut, "spin_mode": step.magnetic_mode,
            }, llm_path)
            with resource_lock:
                if read_resources(folder).revision != previous.revision:
                    raise HTTPException(409, "资源设置已更改，请重新检查。")
                if profile_fingerprint(load_execution_profile()[0]) != profile_fingerprint(profile):
                    raise HTTPException(409, "集群配置已更改，请重新检查资源。")
                selection["profile_sha256"] = profile_fingerprint(profile)
                write_resources(folder, previous.model_copy(update={
                    "selection": selection, "revision": previous.revision + 1,
                }))
            return project_detail(item_id)
        except ResourceSelectionError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            with resource_lock:
                selecting_projects.discard(item_id)

    @app.post("/api/projects/{item_id}/review")
    def review_project(item_id: str, body: ReviewRequest) -> dict[str, Any]:
        folder = _owned_directory(projects_root, item_id)
        bundle = _owned_bundle(folder)
        reviewer = _human_reviewer(body.reviewer)
        with review_lock:
            try:
                inspection = inspect_bundle(bundle)
                if not inspection["integrity_valid"]:
                    raise ValueError("input bundle integrity check failed")
                card = build_review_card(bundle, log_path=review_log)
                if card.status != "pending":
                    raise HTTPException(409, "this input bundle was already reviewed")
                record_input_review(bundle, reviewer=reviewer,
                                    decision=body.decision, notes=body.notes, log_path=review_log)
                return project_detail(item_id)
            except (ValueError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/agent/sessions", status_code=202)
    def create_session(body: AgentRequest, background: BackgroundTasks) -> dict[str, Any]:
        if not llm_path.is_file():
            raise HTTPException(503, "LLM configuration is not available")
        item_id = _new_id()
        goal = body.goal.strip()
        if not goal:
            raise HTTPException(422, "goal must not be blank")
        with session_lock:
            if sum(item.status == "running" for item in sessions.values()) >= 4:
                raise HTTPException(429, "too many active agent sessions")
            while len(sessions) >= _MAX_SESSIONS:
                oldest_completed = next(
                    (key for key, item in sessions.items() if item.status == "completed"), None
                )
                if oldest_completed is None:
                    raise HTTPException(429, "agent session history is full")
                del sessions[oldest_completed]
            session = _AgentSession(id=item_id, goal=goal)
            if body.project_id is not None:
                folder = _owned_directory(projects_root, body.project_id)
                with resource_lock:
                    resources = read_resources(folder)
                    if resources.selection is None:
                        raise HTTPException(409, "请先检查或选择该项目的节点与核数。")
                    try:
                        request = ExecutionRequest.model_validate(resources.selection["execution_request"])
                    except (KeyError, ValueError, TypeError) as exc:
                        raise HTTPException(422, "项目的资源选择无效，请重新选择。") from exc
                    profile, _ = load_execution_profile()
                    if profile is None or request.node.casefold() == "mu01" or request.node not in profile.allowed_nodes:
                        raise HTTPException(422, "已选节点不在当前配置中，请重新选择。")
                    if resources.selection.get("profile_sha256") != profile_fingerprint(profile):
                        raise HTTPException(409, "集群配置已更改，请重新检查资源。")
                    if resources.mode == "manual" and (
                        request.node != resources.node or request.mpi_ranks != resources.mpi_ranks
                    ):
                        raise HTTPException(422, "资源方案与手动设置不一致，请重新检查。")
                    bundle = _owned_bundle(folder)
                    card = build_review_card(bundle, log_path=review_log)
                    if not card.integrity_valid or request.step_id not in {step.step_id for step in card.steps}:
                        raise HTTPException(422, "项目输入发生变化，请重新生成方案。")
                    plans_root = folder / "agent-plans"
                    if plans_root.is_symlink():
                        raise HTTPException(422, "project planning folder must not be a link")
                    work_dir = plans_root / item_id
                    work_dir.mkdir(parents=True, mode=0o700)
                    request_path = work_dir / "execution.request.json"
                    request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
                    session.project_id = body.project_id
                    session.resource_revision = resources.revision
                    session.resource_request = request
                    session.request_path = request_path
                    session.goal += (
                        "\n\n项目资源已固定，请先检查输入审核卡，再对固定节点探测并生成资源及执行计划。"
                        "这个会话只生成计划；列出未完成的输入审核和资源授权要求后停止。"
                        "禁止创建新的输入、修改节点/核数，禁止资源批准或提交作业。"
                        "如输入尚未接受，展示审核要求后停止。\n"
                        f"bundle={bundle}\nprofile={execution_profile_path}\nrequest={request_path}\n"
                        f"snapshot={work_dir / 'node.probe.json'}\n"
                        f"resource_plan_output={work_dir / 'node.resource-plan.json'}\n"
                        f"node={request.node}; mpi_ranks={request.mpi_ranks}; step_id={request.step_id}\n"
                    )
            sessions[item_id] = session
        background.add_task(run_session, item_id)
        return session_detail(item_id)

    @app.get("/api/agent/sessions/{item_id}")
    def get_session(item_id: str) -> dict[str, Any]:
        if _ID.fullmatch(item_id) is None:
            raise HTTPException(404, "agent session not found")
        return session_detail(item_id)

    @app.post("/api/agent/sessions/{item_id}/decisions", status_code=202)
    def decide_session(item_id: str, body: DecisionRequest, background: BackgroundTasks) -> dict[str, Any]:
        if _ID.fullmatch(item_id) is None:
            raise HTTPException(404, "agent session not found")
        reviewer = _human_reviewer(body.reviewer)
        with session_lock:
            session = sessions.get(item_id)
            if session is None:
                raise HTTPException(404, "agent session not found")
            if session.status != "pending_approval" or session.result is None or session.graph is None:
                raise HTTPException(409, "this session has no pending approval")
            pending_ids = {item.call_id for item in session.result.pending_approvals}
            if set(body.decisions) != pending_ids:
                raise HTTPException(422, "decisions must cover exactly the pending calls")
            session.status = "running"
        decisions = {call_id: {"approve": approved, "approver": reviewer}
                     for call_id, approved in body.decisions.items()}
        background.add_task(resume_session, item_id, decisions)
        return session_detail(item_id)

    @app.get("/api/results")
    def list_results() -> dict[str, Any]:
        items = []
        for folder in sorted(results_root.iterdir(), key=lambda path: path.name, reverse=True):
            if not _ID.fullmatch(folder.name) or folder.is_symlink() or not (folder / "summary.json").is_file():
                continue
            detail = result_detail(folder.name)
            summary = detail["summary"]
            items.append({"id": folder.name, "converged": summary.get("converged"),
                          "final_energy": summary.get("final_energy"), "plots": detail["plots"]})
        return {"results": items}

    @app.post("/api/results", status_code=201)
    async def import_result(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        if len(files) > len(_RESULT_FILES):
            raise HTTPException(422, "too many result files")
        names = [file.filename or "" for file in files]
        if len(set(names)) != len(names) or any(name not in _RESULT_FILES for name in names):
            raise HTTPException(422, "result filenames must be unique allowed VASP outputs")
        if not {"OSZICAR", "OUTCAR", "vasprun.xml"}.intersection(names):
            raise HTTPException(422, "result needs OSZICAR, OUTCAR or vasprun.xml")
        item_id = _new_id()
        folder = results_root / item_id
        folder.mkdir(mode=0o700)
        total = 0
        complete = False
        try:
            for upload in files:
                total += await _save_upload(upload, folder / upload.filename, _RESULT_FILE_LIMIT)
                if total > _RESULT_TOTAL_LIMIT:
                    raise HTTPException(413, "result upload exceeds total size limit")
            parsed = analyze_calculation(folder)
            plots = {}
            if "TDOS.dat" in names:
                plots["dos"] = plot_tdos(folder / "TDOS.dat", folder / "dos.png").model_dump(mode="json")
            band_name = "BAND.dat" if "BAND.dat" in names else "REFORMATTED_BAND.dat"
            if band_name in names and "KLABELS" in names:
                plots["band"] = plot_band(folder / band_name, folder / "KLABELS", folder / "band.png").model_dump(mode="json")
            (folder / "summary.json").write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
            (folder / "plots.json").write_text(
                json.dumps({"plot_provenance": _PLOT_PROVENANCE if plots else None,
                            "plots": plots}, indent=2),
                encoding="utf-8",
            )
            detail = result_detail(item_id)
            complete = True
            return detail
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            for upload in files:
                await upload.close()
            if not complete:
                _discard_new_directory(results_root, item_id)

    @app.get("/api/results/{item_id}")
    def get_result(item_id: str) -> dict[str, Any]:
        return result_detail(item_id)

    @app.get("/api/results/{item_id}/plots/{kind}")
    def get_plot(item_id: str, kind: str) -> FileResponse:
        folder = _owned_directory(results_root, item_id)
        filename = _PLOT_FILES.get(kind)
        if filename is None:
            raise HTTPException(404, "plot not found")
        source = folder / filename
        if source.is_symlink() or not source.is_file():
            raise HTTPException(404, "plot not found")
        return FileResponse(source, media_type="image/png")

    static_root = Path(__file__).parent / "web_static"
    if static_root.is_dir():
        app.mount("/static", StaticFiles(directory=static_root), name="static")

    @app.get("/")
    def index() -> FileResponse:
        source = static_root / "index.html"
        if not source.is_file():
            raise HTTPException(503, "web frontend is not installed")
        return FileResponse(source)

    return app
