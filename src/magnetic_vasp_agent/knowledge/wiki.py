from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from magnetic_vasp_agent.knowledge.models import (
    KnowledgeCatalog,
    KnowledgeLocationType,
    KnowledgeSource,
    KnowledgeSourceType,
)
from magnetic_vasp_agent.knowledge.wiki_models import (
    WikiCaptureManifest,
    WikiChunk,
    WikiSourceCapture,
)

ALLOWED_WIKI_HOSTS = ("vasp.at", "www.vasp.at")
WIKI_PAGE_PREFIX = "/wiki/index.php/"
WIKI_API_URL = "https://vasp.at/wiki/api.php"
PAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_+.-]{1,100}$")
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_CHUNK_CHARS = 1800
FETCH_TIMEOUT_SECONDS = 20
USER_AGENT = "magnetic-vasp-agent/0.1 local-knowledge-capture"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_https_url(url: str, *, api: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("VASP Wiki capture requires HTTPS")
    if parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise ValueError("VASP Wiki URL cannot contain credentials or a custom port")
    if (parsed.hostname or "").lower() not in ALLOWED_WIKI_HOSTS:
        raise ValueError("VASP Wiki URL host is not allowlisted")
    if parsed.fragment:
        raise ValueError("VASP Wiki capture URL cannot contain a fragment")
    if api:
        if parsed.path != "/wiki/api.php":
            raise ValueError("MediaWiki API redirect left the fixed API path")
    elif parsed.query or not parsed.path.startswith(WIKI_PAGE_PREFIX):
        raise ValueError("source must be a canonical VASP Wiki page URL")


def page_name_from_source(source: KnowledgeSource) -> str:
    if source.source_type != KnowledgeSourceType.OFFICIAL:
        raise ValueError("VASP Wiki capture accepts only official sources")
    if source.location_type != KnowledgeLocationType.EXTERNAL_URL or source.url is None:
        raise ValueError("VASP Wiki capture requires an external URL source")
    url = str(source.url)
    _validate_https_url(url)
    path = urlparse(url).path
    encoded_name = path[len(WIKI_PAGE_PREFIX) :]
    if not encoded_name or "/" in encoded_name:
        raise ValueError("VASP Wiki source must identify exactly one page")
    page_name = unquote(encoded_name)
    if not PAGE_NAME_PATTERN.fullmatch(page_name):
        raise ValueError("VASP Wiki page name contains unsupported characters")
    return page_name


def api_url_for_page(page_name: str) -> str:
    if not PAGE_NAME_PATTERN.fullmatch(page_name):
        raise ValueError("invalid VASP Wiki page name")
    query = urlencode(
        {
            "action": "parse",
            "page": page_name,
            "prop": "text|displaytitle|revid",
            "format": "json",
            "formatversion": "2",
        }
    )
    return f"{WIKI_API_URL}?{query}"


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_https_url(newurl, api=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_wiki_api(api_url: str) -> bytes:
    _validate_https_url(api_url, api=True)
    request = Request(
        api_url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        method="GET",
    )
    opener = build_opener(_SafeRedirectHandler())
    try:
        with opener.open(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            _validate_https_url(response.geturl(), api=True)
            content_type = response.headers.get_content_type()
            if content_type not in {"application/json", "text/json"}:
                raise ValueError(f"unexpected VASP Wiki content type: {content_type}")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_RESPONSE_BYTES:
                raise ValueError("VASP Wiki response exceeds the size limit")
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValueError("VASP Wiki request failed") from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("VASP Wiki response exceeds the size limit")
    return payload


@dataclass(frozen=True)
class _TextBlock:
    section_path: tuple[str, ...]
    text: str


class _WikiHtmlParser(HTMLParser):
    BLOCK_TAGS: ClassVar[set[str]] = {"p", "li", "pre", "tr", "dd", "dt"}
    HEADING_LEVELS: ClassVar[dict[str, int]] = {"h2": 2, "h3": 3, "h4": 4}
    SKIP_TAGS: ClassVar[set[str]] = {"script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[_TextBlock] = []
        self._section_stack: list[tuple[int, str]] = []
        self._heading_tag: str | None = None
        self._heading_parts: list[str] = []
        self._block_tag: str | None = None
        self._block_parts: list[str] = []
        self._skip_depth = 0

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        values = next((value for key, value in attrs if key == "class"), None)
        return set((values or "").split())

    @staticmethod
    def _normalize(parts: list[str]) -> str:
        return re.sub(r"\s+", " ", "".join(parts)).strip()

    @property
    def section_path(self) -> tuple[str, ...]:
        return tuple(title for _, title in self._section_stack)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        classes = self._classes(attrs)
        if self._skip_depth:
            self._skip_depth += 1
            return
        if tag in self.SKIP_TAGS or "mw-editsection" in classes:
            self._skip_depth = 1
            return
        if tag in self.HEADING_LEVELS and self._heading_tag is None:
            self._heading_tag = tag
            self._heading_parts = []
        elif tag in self.BLOCK_TAGS and self._block_tag is None:
            self._block_tag = tag
            self._block_parts = []
        elif tag == "br":
            if self._heading_tag is not None:
                self._heading_parts.append(" ")
            if self._block_tag is not None:
                self._block_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == self._heading_tag:
            title = self._normalize(self._heading_parts)
            level = self.HEADING_LEVELS[tag]
            if title:
                while self._section_stack and self._section_stack[-1][0] >= level:
                    self._section_stack.pop()
                self._section_stack.append((level, title))
            self._heading_tag = None
            self._heading_parts = []
        if tag == self._block_tag:
            text = self._normalize(self._block_parts)
            if text:
                block = _TextBlock(self.section_path, text)
                if not self.blocks or self.blocks[-1] != block:
                    self.blocks.append(block)
            self._block_tag = None
            self._block_parts = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._heading_tag is not None:
            self._heading_parts.append(data)
        if self._block_tag is not None:
            self._block_parts.append(data)


@dataclass(frozen=True)
class _ParsedPage:
    title: str
    page_id: int
    revision_id: int
    html: str


def parse_api_payload(payload: bytes) -> _ParsedPage:
    try:
        root = json.loads(payload)
        parsed = root["parse"]
        title = parsed["title"]
        page_id = parsed["pageid"]
        revision_id = parsed["revid"]
        html = parsed["text"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("invalid MediaWiki API response") from exc
    if not isinstance(title, str) or not title.strip():
        raise ValueError("MediaWiki response has no page title")
    if type(page_id) is not int or page_id <= 0:
        raise ValueError("MediaWiki response has an invalid page id")
    if type(revision_id) is not int or revision_id <= 0:
        raise ValueError("MediaWiki response has an invalid revision id")
    if not isinstance(html, str) or not html.strip():
        raise ValueError("MediaWiki response has no page HTML")
    return _ParsedPage(title.strip(), page_id, revision_id, html)


def extract_blocks(html: str) -> list[_TextBlock]:
    parser = _WikiHtmlParser()
    parser.feed(html)
    parser.close()
    if not parser.blocks:
        raise ValueError("VASP Wiki page produced no extractable text")
    return parser.blocks


def _split_oversized(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    remainder = text
    while len(remainder) > limit:
        boundary = max(
            remainder.rfind(". ", 0, limit),
            remainder.rfind("; ", 0, limit),
            remainder.rfind(" ", 0, limit),
        )
        if boundary < limit // 2:
            boundary = limit
        else:
            boundary += 1
        pieces.append(remainder[:boundary].strip())
        remainder = remainder[boundary:].strip()
    if remainder:
        pieces.append(remainder)
    return pieces


def build_chunks(
    source: KnowledgeSource,
    page: _ParsedPage,
    snapshot_sha256: str,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[WikiChunk]:
    if max_chars < 200:
        raise ValueError("max chunk size is too small")
    blocks = extract_blocks(page.html)
    grouped: list[tuple[tuple[str, ...], str]] = []
    current_path: tuple[str, ...] | None = None
    current_text = ""
    for block in blocks:
        for piece in _split_oversized(block.text, max_chars):
            separator = "\n\n" if current_text else ""
            if (
                current_path == block.section_path
                and len(current_text) + len(separator) + len(piece) <= max_chars
            ):
                current_text += separator + piece
            else:
                if current_text and current_path is not None:
                    grouped.append((current_path, current_text))
                current_path = block.section_path
                current_text = piece
    if current_text and current_path is not None:
        grouped.append((current_path, current_text))

    source_url = str(source.url)
    chunks = []
    for index, (section_path, text) in enumerate(grouped):
        text_hash = digest(text.encode("utf-8"))
        chunks.append(
            WikiChunk(
                schema_version=1,
                chunk_id=f"{source.id}--r{page.revision_id}--{index:04d}",
                source_id=source.id,
                source_type=source.source_type,
                authority=source.authority,
                source_validated=source.validated,
                source_url=source_url,
                page_title=page.title,
                page_id=page.page_id,
                revision_id=page.revision_id,
                section_path=list(section_path),
                chunk_index=index,
                text=text,
                text_sha256=text_hash,
                snapshot_sha256=snapshot_sha256,
                tasks=source.tasks,
                magnetic_modes=source.magnetic_modes,
            )
        )
    if not chunks:
        raise ValueError("VASP Wiki page produced no chunks")
    return chunks


def _json_lines(chunks: list[WikiChunk]) -> bytes:
    lines = [
        json.dumps(
            chunk.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for chunk in chunks
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass(frozen=True)
class _PreparedCapture:
    source: KnowledgeSource
    api_url: str
    raw: bytes
    page: _ParsedPage
    chunks: list[WikiChunk]
    chunk_bytes: bytes


def _prepare_sources(
    catalog: KnowledgeCatalog,
    fetcher: Callable[[str], bytes],
    delay_seconds: float,
) -> list[_PreparedCapture]:
    sources = [
        source
        for source in catalog.sources
        if source.source_type == KnowledgeSourceType.OFFICIAL
        and source.location_type == KnowledgeLocationType.EXTERNAL_URL
    ]
    if not sources:
        raise ValueError("catalog contains no official VASP Wiki sources")
    prepared = []
    for index, source in enumerate(sources):
        page_name = page_name_from_source(source)
        api_url = api_url_for_page(page_name)
        raw = fetcher(api_url)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("VASP Wiki response exceeds the size limit")
        page = parse_api_payload(raw)
        if page.title.casefold() != page_name.replace("_", " ").casefold():
            raise ValueError(f"MediaWiki returned unexpected page {page.title!r}")
        snapshot_hash = digest(raw)
        chunks = build_chunks(source, page, snapshot_hash)
        prepared.append(
            _PreparedCapture(
                source=source,
                api_url=api_url,
                raw=raw,
                page=page,
                chunks=chunks,
                chunk_bytes=_json_lines(chunks),
            )
        )
        if delay_seconds and index + 1 < len(sources):
            time.sleep(delay_seconds)
    return prepared


def capture_wiki(
    catalog: KnowledgeCatalog,
    catalog_payload: bytes,
    output: Path,
    *,
    fetcher: Callable[[str], bytes] = fetch_wiki_api,
    fetched_at: datetime | None = None,
    delay_seconds: float = 0.25,
) -> WikiCaptureManifest:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Wiki capture output exists; choose a new directory")
    prepared = _prepare_sources(catalog, fetcher, delay_seconds)
    observed_at = fetched_at or datetime.now(UTC)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("fetched_at must be timezone-aware")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".magvasp-wiki-stage-", dir=output.parent))
    captures: list[WikiSourceCapture] = []
    try:
        for item in prepared:
            source_dir = staging / item.source.id
            source_dir.mkdir()
            raw_path = source_dir / "api-response.json"
            chunks_path = source_dir / "chunks.jsonl"
            raw_path.write_bytes(item.raw)
            chunks_path.write_bytes(item.chunk_bytes)
            captures.append(
                WikiSourceCapture(
                    source_id=item.source.id,
                    source_url=str(item.source.url),
                    api_url=item.api_url,
                    page_title=item.page.title,
                    page_id=item.page.page_id,
                    revision_id=item.page.revision_id,
                    fetched_at=observed_at,
                    raw_relative_path=raw_path.relative_to(staging).as_posix(),
                    raw_size_bytes=len(item.raw),
                    raw_sha256=digest(item.raw),
                    chunks_relative_path=chunks_path.relative_to(staging).as_posix(),
                    chunks_size_bytes=len(item.chunk_bytes),
                    chunks_sha256=digest(item.chunk_bytes),
                    chunk_count=len(item.chunks),
                )
            )
        manifest = WikiCaptureManifest(
            schema_version=1,
            capture_kind="vasp_wiki_mediawiki_api",
            catalog_sha256=digest(catalog_payload),
            allowed_hosts=list(ALLOWED_WIKI_HOSTS),
            source_count=len(captures),
            sources=captures,
        )
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        staging.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _capture_file(root: Path, relative_path: str, expected: str) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or relative_path != expected
    ):
        raise ValueError("Wiki manifest contains an unsafe file path")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Wiki capture file is missing or linked: {relative_path}")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("Wiki capture file escapes the capture directory") from exc
    return path


def verify_wiki_capture(
    root: Path,
    *,
    catalog_payload: bytes | None = None,
) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Wiki capture root must be a real directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Wiki capture manifest is missing or linked")
    manifest = WikiCaptureManifest.model_validate_json(manifest_path.read_bytes())
    if catalog_payload is not None and manifest.catalog_sha256 != digest(catalog_payload):
        raise ValueError("Wiki capture catalog hash does not match the supplied catalog")

    expected_files = {"manifest.json"}
    total_chunks = 0
    for source in manifest.sources:
        _validate_https_url(source.source_url)
        _validate_https_url(source.api_url, api=True)
        raw_expected = f"{source.source_id}/api-response.json"
        chunks_expected = f"{source.source_id}/chunks.jsonl"
        raw_path = _capture_file(root, source.raw_relative_path, raw_expected)
        chunks_path = _capture_file(root, source.chunks_relative_path, chunks_expected)
        expected_files.update({raw_expected, chunks_expected})

        raw = raw_path.read_bytes()
        chunk_bytes = chunks_path.read_bytes()
        if len(raw) != source.raw_size_bytes or digest(raw) != source.raw_sha256:
            raise ValueError(f"Wiki raw snapshot hash mismatch: {source.source_id}")
        if (
            len(chunk_bytes) != source.chunks_size_bytes
            or digest(chunk_bytes) != source.chunks_sha256
        ):
            raise ValueError(f"Wiki chunk file hash mismatch: {source.source_id}")
        page = parse_api_payload(raw)
        if (
            page.title != source.page_title
            or page.page_id != source.page_id
            or page.revision_id != source.revision_id
        ):
            raise ValueError(f"Wiki snapshot metadata mismatch: {source.source_id}")

        lines = chunk_bytes.decode("utf-8").splitlines()
        if len(lines) != source.chunk_count:
            raise ValueError(f"Wiki chunk count mismatch: {source.source_id}")
        for index, line in enumerate(lines):
            chunk = WikiChunk.model_validate_json(line)
            if (
                chunk.source_id != source.source_id
                or chunk.page_title != source.page_title
                or chunk.page_id != source.page_id
                or chunk.revision_id != source.revision_id
                or chunk.chunk_index != index
                or chunk.snapshot_sha256 != source.raw_sha256
                or chunk.text_sha256 != digest(chunk.text.encode("utf-8"))
            ):
                raise ValueError(f"Wiki chunk provenance mismatch: {chunk.chunk_id}")
        total_chunks += len(lines)

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("Wiki capture contains missing, extra, or linked files")
    return {
        "valid": True,
        "source_count": manifest.source_count,
        "chunk_count": total_chunks,
        "catalog_sha256": manifest.catalog_sha256,
    }
