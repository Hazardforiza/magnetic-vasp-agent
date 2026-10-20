"""Reviewed local/shared library with atomic, filesystem-backed BM25 snapshots.

Only an explicitly configured directory is shared. No document text is sent to an
embedding provider, and neither importing nor indexing certifies scientific facts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from magnetic_vasp_agent.knowledge.models import KnowledgeSourceType
from magnetic_vasp_agent.knowledge.retrieval import (
    build_document_index,
    search_bm25,
    verify_bm25_index,
)
from magnetic_vasp_agent.knowledge.retrieval_models import RetrievalDocument

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 1_000_000
MAX_LIBRARY_CHARS = 12_000_000
MAX_STATE_BYTES = 64 * 1024 * 1024
MAX_ITEMS = 500
_ID = re.compile(r"^[0-9a-f]{32}$")
_RAW_NAMES = {"OUTCAR", "OSZICAR", "WAVECAR", "CHGCAR", "CHG", "EIGENVAL", "DOSCAR", "PROCAR", "VASPRUN.XML"}
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class LibraryError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _check_path(path: Path) -> None:
    if any(_linked(part) for part in [path, *path.parents]):
        raise LibraryError("知识库目录不能包含符号链接或目录联接。")


def _read_json(path: Path, limit: int = 64 * 1024 * 1024) -> Any:
    if _linked(path) or not path.is_file() or path.stat().st_size > limit:
        raise LibraryError("知识库文件缺失、过大或包含链接。", 503)
    try:
        return json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise LibraryError("知识库文件损坏，请检查存储目录。", 503) from exc


def _atomic_json(path: Path, value: Any) -> None:
    if _linked(path):
        raise LibraryError("知识库文件不能是链接。")
    temporary = path.parent / f".write-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _contains_links(root: Path) -> bool:
    for current, directories, files in os.walk(root, followlinks=False):
        if any(_linked(Path(current) / name) for name in [*directories, *files]):
            return True
    return False


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    """OS byte/flock locking also works between app instances on shared storage.

    The network filesystem must support normal OS file locks and atomic rename.
    A crashed process releases its lock; no stale lock deletion is necessary.
    """
    _check_path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = str(root.resolve()).casefold() if os.name == "nt" else str(root.resolve())
    with _THREAD_LOCKS_GUARD:
        guard = _THREAD_LOCKS.setdefault(key, threading.RLock())
    if not guard.acquire(timeout=15):
        raise LibraryError("知识库正在更新，请稍后重试。", 409)
    lock_path = root / ".library.lock"
    handle = None
    try:
        if _linked(lock_path):
            raise LibraryError("知识库锁文件不能是链接。")
        handle = lock_path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + 15
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise LibraryError("知识库正被其他电脑更新，请稍后重试。", 409) from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise LibraryError("无法访问知识库目录，请检查共享路径及读写权限。", 503) from exc
    finally:
        if handle is not None:
            handle.close()
        guard.release()


def _clean_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LibraryError("正文不能为空。")
    if len(value) > MAX_TEXT_CHARS:
        raise LibraryError("正文超过 100 万字符限制。", 413)
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise LibraryError("正文包含二进制或不可读字符。")
    if re.search(r"^\s*TITEL\s*=", value, re.MULTILINE) and re.search(r"VRHFIN|End of Dataset|PAW_PBE|PAW_LDA", value):
        raise LibraryError("不能将 POTCAR 原文加入知识库；请记录赝势名称和参数摘要。")
    return value.strip()


def _metadata(title: str, doi: str | None, tags: list[str] | None) -> dict[str, Any]:
    title = title.strip()
    if not title or len(title) > 300:
        raise LibraryError("标题须为 1–300 个字符。")
    doi = doi.strip() if doi else None
    if doi and (len(doi) > 300 or any(ord(c) < 32 for c in doi)):
        raise LibraryError("DOI 格式或长度无效。")
    tags = list(dict.fromkeys(tag.strip() for tag in (tags or []) if tag.strip()))
    if len(tags) > 20 or any(len(tag) > 60 for tag in tags):
        raise LibraryError("最多添加 20 个标签，每个标签不超过 60 字。")
    return {"title": title, "doi": doi, "tags": tags}


def _source_provenance(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item["provenance"].items()
            if key not in {"published_from", "published_at"}}


def _content_identity(item: dict[str, Any]) -> str:
    identity = {"kind": item["kind"], "text": item["text"]}
    if item["kind"] == "case":
        identity["provenance"] = _source_provenance(item)
    return hashlib.sha256(_json(identity)).hexdigest()


def _source_segments(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep editable notes separate from immutable application-captured evidence."""
    if item["kind"] == "case":
        segments = [{"key": "notes", "page": None, "text": item["text"], "label": "笔记"}]
        for source, label in (("project", "项目"), ("result", "结果")):
            snapshot = item["provenance"].get(source)
            if isinstance(snapshot, dict):
                segments.append({"key": source + "_snapshot", "page": None,
                                 "text": json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2),
                                 "label": "关联来源快照 · " + label})
        return segments
    pages = [{"key": "body", "page": page["page"], "text": page["text"],
              "label": f"PDF 第 {page['page']} 页"} for page in item["pages"] if page["text"].strip()]
    return pages or [{"key": "body", "page": None, "text": item["text"], "label": "正文"}]


# Extraction runs outside the web process: malformed/compressed PDFs cannot keep
# a request thread executing indefinitely. The worker has no networking or files.
_PDF_WORKER = r'''
import io,json,sys
sys.stdout.reconfigure(encoding="utf-8")
try:
    if sys.platform != "win32":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (768*1024*1024,768*1024*1024))
    from pypdf import PdfReader
    reader=PdfReader(io.BytesIO(sys.stdin.buffer.read()), strict=True)
    if reader.is_encrypted:
        raise ValueError("PDF 已加密，请先解密后再导入。")
    if len(reader.pages)>400:
        raise ValueError("PDF 超过 400 页限制，请拆分文献。")
    pages=[];total=0
    for number,page in enumerate(reader.pages,1):
        contents=page.get_contents()
        if contents is not None and len(contents.get_data())>16*1024*1024:
            raise ValueError("PDF 页面解压后过大，请先简化文档。")
        text=(page.extract_text() or "").strip()
        total+=len(text)
        if total>1000000:
            raise ValueError("PDF 提取正文超过 100 万字符限制。")
        pages.append({"page":number,"text":text})
    if not any(page["text"] for page in pages):
        raise ValueError("PDF 没有可提取文字，可能是扫描件；请先 OCR 后导入。")
    sys.stdout.write(json.dumps({"pages":pages},ensure_ascii=False))
except ImportError:
    sys.stdout.write(json.dumps({"error":"缺少 PDF 解析依赖，请安装 web 运行环境。"}))
except Exception as exc:
    message=str(exc) if isinstance(exc,ValueError) else "PDF 无法解析，请确认文件有效或改用文字版。"
    sys.stdout.write(json.dumps({"error":message},ensure_ascii=False))
'''


def _extract(filename: str, payload: bytes) -> tuple[str, list[dict[str, Any]]]:
    if (not filename or len(filename) > 240 or Path(filename).name != filename
            or any(char in filename for char in "/\\:\0")):
        raise LibraryError("上传文件名无效。")
    name = filename.upper()
    if "POTCAR" in name or name in _RAW_NAMES or any(name.startswith(raw + ".") for raw in _RAW_NAMES):
        raise LibraryError("请导入文献或结构化案例，不支持 POTCAR 和 VASP 原始输出。")
    if not payload:
        raise LibraryError("上传文件为空。")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise LibraryError("上传文件超过 20 MiB 限制。", 413)
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        if not payload.startswith(b"%PDF-"):
            raise LibraryError("文件不是有效 PDF。")
        try:
            result = subprocess.run([sys.executable, "-c", _PDF_WORKER], input=payload,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=25, check=False)
            extracted = json.loads(result.stdout.decode("utf-8"))
        except (subprocess.TimeoutExpired, ValueError, UnicodeError) as exc:
            raise LibraryError("PDF 解析超时或失败，请拆分文件或改用文字版。") from exc
        if extracted.get("error"):
            raise LibraryError(extracted["error"])
        pages = extracted["pages"]
        text = _clean_text("\n\n".join(page["text"] for page in pages))
        return text, pages
    if suffix not in {".txt", ".md", ".bib"}:
        raise LibraryError("支持 PDF、Markdown、UTF-8 TXT 和 BibTeX 文件。")
    try:
        return _clean_text(payload.decode("utf-8-sig")), []
    except UnicodeError as exc:
        raise LibraryError("文字文件须使用 UTF-8 编码。") from exc


class KnowledgeLibrary:
    def __init__(self, data_root: Path):
        self.base = Path(data_root).absolute() / "knowledge"
        _check_path(self.base)
        self.base.mkdir(parents=True, exist_ok=True)
        self.local = self.base / "local"
        self.settings = self.base / "settings.json"

    def _configured_path(self) -> Path | None:
        if not self.settings.exists():
            return None
        settings = _read_json(self.settings, 16384)
        value = settings.get("shared_path")
        return Path(value) if value else None

    def _root(self, scope: str) -> Path:
        if scope == "local":
            _check_path(self.local)
            return self.local
        if scope != "shared":
            raise LibraryError("知识库范围必须为 local 或 shared。")
        configured = self._configured_path()
        if configured is None:
            raise LibraryError("共享库尚未配置，请选择组内共享目录。", 503)
        _check_path(configured)
        if not configured.is_dir():
            raise LibraryError("共享目录不可访问，请连接组内网络并检查路径。", 503)
        return configured / "magvasp-knowledge"

    def configure_shared(self, path: str | None) -> dict[str, Any]:
        if path and path.strip():
            shared = Path(path.strip())
            if not shared.is_absolute():
                raise LibraryError("共享路径必须为绝对路径或网络共享路径。")
            _check_path(shared)
            if not shared.is_dir():
                raise LibraryError("共享目录不存在或无法访问，请先挂载组内共享目录。")
            resolved = shared.resolve()
            local = self.base.resolve()
            if resolved == local or resolved in local.parents or local in resolved.parents:
                raise LibraryError("共享目录不能与本机知识库目录重叠。")
            root = shared / "magvasp-knowledge"
            with _lock(root):
                self._load(root)
            value = str(shared)
        else:
            value = None
        with _lock(self.base):
            _atomic_json(self.settings, {"schema_version": 1, "shared_path": value})
        return self.overview()

    def _load(self, root: Path) -> tuple[dict[str, dict[str, Any]], Path | None]:
        pointer = root / "current.json"
        if not pointer.exists():
            return {}, None
        current = _read_json(pointer, 4096)
        generation = current.get("generation", "")
        if not isinstance(generation, str) or not _ID.fullmatch(generation):
            raise LibraryError("知识库索引指针无效。", 503)
        snapshot = root / "snapshots" / generation
        _check_path(snapshot)
        state = _read_json(snapshot / "items.json")
        if state.get("schema_version") != 1 or not isinstance(state.get("items"), list):
            raise LibraryError("知识库版本或内容无效。", 503)
        items = {}
        if len(state["items"]) > MAX_ITEMS:
            raise LibraryError("知识库条目超过限制。", 503)
        for item in state["items"]:
            if (not isinstance(item, dict) or not _ID.fullmatch(str(item.get("id", "")))
                    or item.get("status") not in {"draft", "indexed"}
                    or item.get("validation_status") != "unverified"
                    or item.get("kind") not in {"literature", "case"}
                    or not isinstance(item.get("text"), str) or not item["text"].strip()
                    or len(item["text"]) > MAX_TEXT_CHARS
                    or not isinstance(item.get("title"), str) or not 1 <= len(item["title"]) <= 300
                    or not isinstance(item.get("created_at"), str)
                    or not isinstance(item.get("provenance"), dict)
                    or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("content_sha256", "")))
                    or not isinstance(item.get("pages"), list)
                    or len(item["pages"]) > 400 or item.get("page_count") != len(item["pages"])
                    or not isinstance(item.get("tags"), list)
                    or len(item["tags"]) > 20 or any(not isinstance(tag, str) or len(tag) > 60 for tag in item["tags"])
                    or "doi" not in item or (item["doi"] is not None and not isinstance(item["doi"], str))
                    or any(not isinstance(page, dict) or page.get("page") != number
                           or not isinstance(page.get("text"), str) or len(page["text"]) > MAX_TEXT_CHARS
                           for number, page in enumerate(item["pages"], 1))
                    or item["id"] in items):
                raise LibraryError("知识库条目无效。", 503)
            items[item["id"]] = item
        return items, snapshot

    def _commit(self, root: Path, items: dict[str, dict[str, Any]]) -> None:
        if len(items) > MAX_ITEMS or sum(len(item["text"]) for item in items.values()) > MAX_LIBRARY_CHARS:
            raise LibraryError("知识库超过 500 条或 1200 万字符容量限制。", 413)
        documents: list[RetrievalDocument] = []
        for item in items.values():
            if item["status"] != "indexed":
                continue
            for segment in _source_segments(item):
                page, text = segment["page"], segment["text"]
                # Small overlapping passages produce bounded, usable quotations.
                for start in range(0, len(text), 1000):
                    chunk = text[start:start + 1200].strip()
                    if not chunk:
                        continue
                    metadata = (f"标题：{item['title']}\nDOI：{item['doi'] or ''}\n"
                                f"标签：{', '.join(item['tags'])}\n\n")
                    indexed_text = metadata + chunk
                    documents.append(RetrievalDocument(
                        schema_version=1, document_id=f"{item['id']}-{segment['key']}-{page or 0}-{start}",
                        source_id=item["id"], source_type=KnowledgeSourceType.PERSONAL_NOTE,
                        authority=30, source_validated=False, title=item["title"],
                        section_path=[segment["label"]], text=indexed_text,
                        text_sha256=hashlib.sha256(indexed_text.encode("utf-8")).hexdigest(),
                        tasks=[], magnetic_modes=[],
                    ))
        if len(documents) > 10000:
            raise LibraryError("知识库超过 10000 个检索片段限制，请拆分资料库。", 413)
        snapshots = root / "snapshots"
        _check_path(snapshots)
        snapshots.mkdir(exist_ok=True)
        generation = uuid.uuid4().hex
        snapshot = snapshots / generation
        snapshot.mkdir()
        payload = _json({"schema_version": 1, "items": list(items.values())})
        try:
            if len(payload) > MAX_STATE_BYTES:
                raise LibraryError("知识库内容超过 64 MiB 存储容量限制，请拆分资料库。", 413)
            (snapshot / "items.json").write_bytes(payload)
            if documents:
                build_document_index(documents, snapshot / "index", corpus_payload=payload)
            _atomic_json(root / "current.json", {"schema_version": 1, "generation": generation})
        except Exception:
            shutil.rmtree(snapshot, ignore_errors=True)
            raise
        # Readers hold the same OS lock, so retired generations are safe to remove.
        try:
            for old in snapshots.iterdir():
                if (old.name != generation and _ID.fullmatch(old.name) and old.is_dir()
                        and not _linked(old) and not _contains_links(old)):
                    shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass  # The active snapshot is already committed; retirement is optional.

    def _verify_active_index(self, items: dict[str, dict[str, Any]], snapshot: Path | None) -> bool:
        if not any(item["status"] == "indexed" for item in items.values()):
            return True
        if snapshot is None:
            raise LibraryError("已入库资料的检索索引缺失，请重新确认入库。", 503)
        try:
            _check_path(snapshot / "index")
            manifest = verify_bm25_index(snapshot / "index")
            if manifest.catalog_sha256 != hashlib.sha256((snapshot / "items.json").read_bytes()).hexdigest():
                raise ValueError("state/index mismatch")
        except (OSError, ValueError) as exc:
            raise LibraryError("检索索引缺失或损坏，请重新确认入库以重建索引。", 503) from exc
        return True

    @staticmethod
    def _public(item: dict[str, Any], scope: str, *, summary: bool = False) -> dict[str, Any]:
        result = {key: value for key, value in item.items() if key != "content_sha256"}
        result["scope"] = scope
        if summary:
            for key in ("text", "pages", "provenance"):
                result.pop(key, None)
        return result

    def overview(self) -> dict[str, Any]:
        result: dict[str, Any] = {"items": []}
        for scope in ("local", "shared"):
            status = {"scope": scope, "label": "本地库" if scope == "local" else "共享库",
                      "configured": scope == "local", "available": False, "path": None,
                      "item_count": 0, "indexed_count": 0, "index_ready": False, "error": None}
            try:
                configured = self._configured_path() if scope == "shared" else self.local
                status["configured"] = configured is not None
                status["path"] = str(configured) if configured else None
                if configured is not None:
                    root = self._root(scope)
                    with _lock(root):
                        items, snapshot = self._load(root)
                        status["item_count"] = len(items)
                        result["items"].extend(self._public(item, scope, summary=True) for item in items.values())
                        self._verify_active_index(items, snapshot)
                    status.update(available=True, item_count=len(items),
                                  indexed_count=sum(item["status"] == "indexed" for item in items.values()), index_ready=True)
            except (LibraryError, OSError) as exc:
                status["error"] = str(exc) if isinstance(exc, LibraryError) else "无法访问知识库目录。"
            result[scope] = status
        result["items"].sort(key=lambda item: item["created_at"], reverse=True)
        return result

    def get(self, scope: str, item_id: str) -> dict[str, Any]:
        if not _ID.fullmatch(item_id):
            raise LibraryError("知识库条目不存在。", 404)
        root = self._root(scope)
        with _lock(root):
            items, _ = self._load(root)
            if item_id not in items:
                raise LibraryError("知识库条目不存在。", 404)
            return self._public(items[item_id], scope)

    def _insert(self, scope: str, item: dict[str, Any], *, expected_root: Path | None = None) -> dict[str, Any]:
        root = self._root(scope)
        if expected_root is not None and root.resolve() != expected_root.resolve():
            raise LibraryError("共享库配置在本次操作期间发生变化，请重新提交。", 409)
        with _lock(root):
            items, _ = self._load(root)
            for existing in items.values():
                # Always compare the current body, including snapshots created
                # by older versions whose ingestion digest survived an edit.
                # A matching original-upload hash must never confirm edited text.
                if (existing["kind"] == item["kind"] and existing["text"] == item["text"]
                        and (item["kind"] == "literature"
                             or _source_provenance(existing) == _source_provenance(item))):
                    # Publishing is an explicit confirmation even when the same
                    # document was previously imported as a shared draft.
                    if item["status"] == "indexed" and existing["status"] == "draft":
                        if any(existing[key] != item[key] for key in ("title", "doi", "tags")):
                            continue  # Differing shared edits require their own confirmation.
                        existing["status"] = "indexed"
                        self._commit(root, items)
                    return {**self._public(existing, scope), "duplicate": True}
            items[item["id"]] = item
            self._commit(root, items)
            return self._public(item, scope)

    def import_file(self, scope: str, filename: str, payload: bytes, *, title: str | None = None,
                    doi: str | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        expected_root = self._root(scope)  # Capture destination before PDF work.
        text, pages = _extract(filename, payload)
        item = {"id": uuid.uuid4().hex, "kind": "literature",
                **_metadata(title or Path(filename).stem, doi, tags), "filename": filename,
                "status": "draft", "text": text, "pages": pages, "page_count": len(pages),
                "page_mapping": "original_pdf" if pages else "none",
                "created_at": datetime.now(timezone.utc).isoformat(), "provenance": {},
                "validation_status": "unverified", "content_sha256": ""}
        item["content_sha256"] = _content_identity(item)
        return self._insert(scope, item, expected_root=expected_root)

    def create_case(self, scope: str, title: str, text: str, tags: list[str] | None = None,
                    provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        expected_root = self._root(scope)
        text = _clean_text(text)
        item = {"id": uuid.uuid4().hex, "kind": "case", **_metadata(title, None, tags),
                "filename": None, "status": "draft", "text": text, "pages": [], "page_count": 0,
                "page_mapping": "none", "created_at": datetime.now(timezone.utc).isoformat(),
                "provenance": provenance or {}, "validation_status": "unverified",
                "content_sha256": ""}
        item["content_sha256"] = _content_identity(item)
        return self._insert(scope, item, expected_root=expected_root)

    def update(self, scope: str, item_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        if not _ID.fullmatch(item_id):
            raise LibraryError("知识库条目不存在。", 404)
        if not set(changes).issubset({"title", "doi", "tags", "text"}):
            raise LibraryError("不支持修改该字段。")
        root = self._root(scope)
        with _lock(root):
            items, _ = self._load(root)
            if item_id not in items:
                raise LibraryError("知识库条目不存在。", 404)
            item = items[item_id]
            item.update(_metadata(changes.get("title", item["title"]), changes.get("doi", item["doi"]),
                                  changes.get("tags", item["tags"])))
            if "text" in changes:
                text = _clean_text(changes["text"])
                if text != item["text"]:
                    item.update(text=text, pages=[], page_count=0, page_mapping="edited_text")
            item["status"] = "draft"
            item["content_sha256"] = _content_identity(item)
            self._commit(root, items)
            return self._public(item, scope)

    def index(self, scope: str, item_id: str) -> dict[str, Any]:
        if not _ID.fullmatch(item_id):
            raise LibraryError("知识库条目不存在。", 404)
        root = self._root(scope)
        with _lock(root):
            items, _ = self._load(root)
            if item_id not in items:
                raise LibraryError("知识库条目不存在。", 404)
            items[item_id]["status"] = "indexed"
            self._commit(root, items)
            return self._public(items[item_id], scope)

    def publish(self, item_id: str) -> dict[str, Any]:
        expected_root = self._root("shared")
        local = self.get("local", item_id)
        if local["status"] != "indexed":
            raise LibraryError("请先在本地预览并确认入库，再发布至共享库。", 409)
        local.pop("scope")
        local["content_sha256"] = _content_identity(local)
        # A fresh shared ID prevents local/shared path identity from being confused.
        local["id"] = uuid.uuid4().hex
        local["provenance"] = {**local["provenance"], "published_from": item_id,
                               "published_at": datetime.now(timezone.utc).isoformat()}
        return self._insert("shared", local, expected_root=expected_root)

    def search(self, query: str, scope: str = "all", limit: int = 5) -> dict[str, Any]:
        query = query.strip()
        if not query or len(query) > 1000 or scope not in {"all", "local", "shared"} or not 1 <= limit <= 10:
            raise LibraryError("请输入 1–1000 字的检索词，有效范围和 1–10 条结果。")
        evidence = []
        warnings = []
        for current_scope in (["local", "shared"] if scope == "all" else [scope]):
            try:
                if current_scope == "shared" and self._configured_path() is None:
                    if scope == "shared":
                        warnings.append("共享库尚未配置。")
                    continue
                root = self._root(current_scope)
                with _lock(root):
                    items, snapshot = self._load(root)
                    if snapshot is None or not any(item["status"] == "indexed" for item in items.values()):
                        continue
                    self._verify_active_index(items, snapshot)
                    found = search_bm25(snapshot / "index", query, limit=limit)
                    for record in found.evidence:
                        item = items[record.source_id]
                        page = None
                        if record.section_path:
                            match = re.fullmatch(r"PDF 第 (\d+) 页", record.section_path[0])
                            page = int(match[1]) if match else None
                        parts = record.document_id.rsplit("-", 3)
                        start = int(parts[-1])
                        segment_key = parts[1] if len(parts) == 4 else ("notes" if item["kind"] == "case" else "body")
                        segment = next((entry for entry in _source_segments(item)
                                        if entry["key"] == segment_key and entry["page"] == page), None)
                        if segment is None:
                            raise LibraryError("检索片段与来源不一致，请重新确认入库。", 503)
                        source_text = segment["text"]
                        quote = source_text[start:start + 1200].strip()
                        evidence.append({"item_id": item["id"], "scope": current_scope, "title": item["title"],
                                         "quote": quote, "page": page, "doi": item["doi"], "tags": item["tags"],
                                         "kind": item["kind"], "source_type": "personal_note", "source_validated": False,
                                         "validation_status": "unverified", "score": record.score,
                                         "source_segment": segment["key"], "source_segment_label": segment["label"],
                                         "section_path": record.section_path,
                                         "document_id": record.document_id,
                                         "text_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest()})
            except (LibraryError, OSError, ValueError, KeyError) as exc:
                warnings.append(f"{'共享库' if current_scope == 'shared' else '本地库'}检索暂不可用："
                                + (str(exc) if isinstance(exc, LibraryError) else "索引不可读取，请检查存储目录。"))
        evidence.sort(key=lambda item: (-item["score"], item["scope"], item["document_id"]))
        for rank, item in enumerate(evidence[:limit], 1):
            item["citation_id"] = f"K{rank}"
        return {"query": query, "evidence": evidence[:limit], "warnings": warnings}
