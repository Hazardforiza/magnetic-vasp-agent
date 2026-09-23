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

from magnetic_vasp_agent.input_review import build_review_card, record_input_review
from magnetic_vasp_agent.input_review_models import REVIEWER_BLOCKLIST
from magnetic_vasp_agent.mcif_workflow import plan_workflow_from_mcif
from magnetic_vasp_agent.models import PotentialPolicy, PotentialRegistry
from magnetic_vasp_agent.postprocess import analyze_calculation
from magnetic_vasp_agent.vaspkit_plot import plot_band, plot_tdos
from magnetic_vasp_agent.workflow import digest, inspect_bundle, materialize_workflow, prepare_workflow
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


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer: str = Field(min_length=1, max_length=64)
    decision: Literal["accept", "reject", "revise"]
    notes: str | None = Field(default=None, max_length=4000)


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    goal: str = Field(min_length=1, max_length=4000)


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
) -> FastAPI:
    """Create a loopback-only app; its paths come from the server operator."""

    data_root = Path(root or "var/web").resolve()
    registry_path = Path(registry or "var/potcar_registry.json").resolve()
    policy_path = Path(policy or "config/potential_policy.json").resolve()
    llm_path = Path(llm_config or "config/agent/llm.json").resolve()
    projects_root = data_root / "projects"
    results_root = data_root / "results"
    review_log = data_root / "reviews" / "input-reviews.jsonl"
    projects_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    csrf_token = secrets.token_urlsafe(32)
    sessions: dict[str, _AgentSession] = {}
    session_lock = threading.RLock()
    review_lock = threading.RLock()
    app = FastAPI(title="MagVASP Local Web", docs_url=None, redoc_url=None, openapi_url=None)

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
        return {"id": item_id, "review_card": card.model_dump(mode="json"), "steps": steps}

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
            from magnetic_vasp_agent.agent.llm_config import load_llm_config

            graph = VaspAgentGraph(load_llm_config(llm_path))
            result = graph.run(sessions[item_id].goal)
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
        }

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
    ) -> dict[str, Any]:
        if Path(file.filename or "").suffix.lower() not in {".mcif", ".cif"}:
            raise HTTPException(422, "upload an .mcif or .cif file")
        if not registry_path.is_file() or not policy_path.is_file():
            raise HTTPException(503, "POTCAR registry or potential policy is not configured")
        item_id = _new_id()
        folder = projects_root / item_id
        folder.mkdir(mode=0o700)
        source = folder / "source.mcif"
        complete = False
        try:
            await _save_upload(file, source, _MCIF_LIMIT)
            request_file = folder / "workflow.request.json"
            plan_workflow_from_mcif(
                [source], request_file, workflow_id=item_id,
                task=task, encut=encut, spacing=spacing, include_dos=include_dos,
                review_log=review_log,
            )
            payloads = {
                "workflow": request_file.read_bytes(),
                "registry": registry_path.read_bytes(),
                "policy": policy_path.read_bytes(),
            }
            prepared = prepare_workflow(
                Workflow.model_validate_json(payloads["workflow"]),
                PotentialRegistry.model_validate_json(payloads["registry"]),
                PotentialPolicy.model_validate_json(payloads["policy"]),
                request_file.parent,
                sources={name: {"sha256": digest(payload)} for name, payload in payloads.items()},
            )
            materialize_workflow(prepared, folder / "bundle")
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
            sessions[item_id] = _AgentSession(id=item_id, goal=body.goal.strip())
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
