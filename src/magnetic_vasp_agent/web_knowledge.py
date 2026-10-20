"""Knowledge-library routes; authorization/CSRF is supplied by the loopback app."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from magnetic_vasp_agent.knowledge.library import (
    MAX_TEXT_CHARS,
    MAX_UPLOAD_BYTES,
    KnowledgeLibrary,
    LibraryError,
)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SharedLibraryRequest(_Request):
    path: str | None = Field(default=None, max_length=2000)


class LibraryEditRequest(_Request):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    doi: str | None = Field(default=None, max_length=300)
    tags: list[str] | None = Field(default=None, max_length=20)
    text: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)


class LibraryCaseRequest(_Request):
    scope: Literal["local", "shared"] = "local"
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    tags: list[str] = Field(default_factory=list, max_length=20)
    project_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    result_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class LibrarySearchRequest(_Request):
    query: str = Field(min_length=1, max_length=1000)
    scope: Literal["all", "local", "shared"] = "all"
    limit: int = Field(default=5, ge=1, le=10)


def _invoke(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except LibraryError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, "无法访问知识库目录，请检查连接和读写权限。") from exc


def create_knowledge_router(
    service: KnowledgeLibrary, *, project_summary: Callable[[str], dict],
    result_summary: Callable[[str], dict],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/knowledge")
    def library_overview() -> dict:
        return _invoke(service.overview)

    @router.put("/api/knowledge/shared")
    def configure_shared(payload: SharedLibraryRequest) -> dict:
        return _invoke(service.configure_shared, payload.path)

    @router.post("/api/knowledge/import")
    async def import_document(
        scope: Literal["local", "shared"] = Form("local"),
        file: UploadFile = File(...), title: str | None = Form(None),
        doi: str | None = Form(None), tags: str = Form(""),
    ) -> dict:
        chunks = []
        size = 0
        try:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "上传文件超过 20 MiB 限制。")
                chunks.append(chunk)
            if len(tags) > 2000:
                raise HTTPException(422, "标签过长。")
            return await asyncio.to_thread(
                _invoke, service.import_file, scope, file.filename or "", b"".join(chunks),
                title=title, doi=doi, tags=tags.replace("，", ",").split(","),
            )
        finally:
            await file.close()

    @router.post("/api/knowledge/cases")
    def create_case(payload: LibraryCaseRequest) -> dict:
        provenance: dict[str, Any] = {"scientifically_verified": False,
                                      "input_review_is_scientific_validation": False}
        if payload.project_id:
            summary = project_summary(payload.project_id)
            provenance["project"] = {"id": payload.project_id, "summary": summary}
        if payload.result_id:
            summary = result_summary(payload.result_id)
            provenance["result"] = {"id": payload.result_id, "summary": summary}
        return _invoke(service.create_case, payload.scope, payload.title, payload.text, payload.tags, provenance)

    @router.post("/api/knowledge/search")
    def search_library(payload: LibrarySearchRequest) -> dict:
        return _invoke(service.search, payload.query, scope=payload.scope, limit=payload.limit)

    @router.get("/api/knowledge/{scope}/{item_id}")
    def get_item(scope: Literal["local", "shared"], item_id: str) -> dict:
        return _invoke(service.get, scope, item_id)

    @router.patch("/api/knowledge/{scope}/{item_id}")
    def edit_item(scope: Literal["local", "shared"], item_id: str, payload: LibraryEditRequest) -> dict:
        changes = payload.model_dump(exclude_unset=True)
        if any(changes.get(field) is None for field in ("title", "tags", "text") if field in changes):
            raise HTTPException(422, "标题、标签和正文不能设为空值。")
        return _invoke(service.update, scope, item_id, changes)

    @router.post("/api/knowledge/{scope}/{item_id}/index")
    def index_item(scope: Literal["local", "shared"], item_id: str) -> dict:
        return _invoke(service.index, scope, item_id)

    @router.post("/api/knowledge/local/{item_id}/publish")
    def publish_item(item_id: str) -> dict:
        return _invoke(service.publish, item_id)

    return router
