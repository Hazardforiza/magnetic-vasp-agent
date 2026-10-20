from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from magnetic_vasp_agent.knowledge.models import (
    ContentHandling,
    DuplicateContentGroup,
    InventoryDocument,
    InventoryStatus,
    InventorySummary,
    KnowledgeCatalog,
    KnowledgeInventory,
    KnowledgeLocationType,
    KnowledgeSource,
    SourceInventory,
)

SKIP_DIRECTORY_NAMES = {
    ".git",
    ".system",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
}

BLOCKED_EXACT_NAMES = {
    "AECCAR0",
    "AECCAR1",
    "AECCAR2",
    "CHG",
    "CHGCAR",
    "CONTCAR",
    "DOSCAR",
    "EIGENVAL",
    "IBZKPT",
    "LOCPOT",
    "OSZICAR",
    "OUTCAR",
    "PROCAR",
    "VASPRUN.XML",
    "WAVECAR",
    "XDATCAR",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_catalog(path: Path) -> tuple[KnowledgeCatalog, bytes]:
    payload = path.read_bytes()
    return KnowledgeCatalog.model_validate_json(payload), payload


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError(f"{label} must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path")
    if ":" in path.parts[0]:
        raise ValueError(f"{label} must not contain a Windows drive")
    return path


def _is_blocked_vasp_artifact(path: Path) -> bool:
    name = path.name.upper()
    return name in BLOCKED_EXACT_NAMES or name.startswith("POTCAR")


def _source_declaration_hash(source: KnowledgeSource) -> str:
    payload = json.dumps(
        source.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return digest(payload)


def _document_id(source_id: str, relative_path: str) -> str:
    short_hash = digest(relative_path.encode("utf-8"))[:16]
    return f"{source_id}--{short_hash}"


def _resolve_base(catalog_path: Path, catalog: KnowledgeCatalog) -> Path:
    if "\\" in catalog.base_directory:
        raise ValueError("catalog base_directory must use forward slashes")
    declared = PurePosixPath(catalog.base_directory)
    if declared.is_absolute() or (declared.parts and ":" in declared.parts[0]):
        raise ValueError("catalog base_directory must be relative")
    candidate = (catalog_path.parent / catalog.base_directory).resolve(strict=True)
    if candidate == Path(candidate.anchor):
        raise ValueError("catalog base_directory cannot be a filesystem root")
    if not candidate.is_dir():
        raise ValueError("catalog base_directory must resolve to a real directory")
    try:
        catalog_path.resolve(strict=True).relative_to(candidate)
    except ValueError as exc:
        raise ValueError("catalog file must be inside base_directory") from exc
    return candidate


def _resolve_source(base: Path, source: KnowledgeSource) -> Path:
    relative = _safe_relative_path(source.path or "", label=f"source {source.id} path")
    unresolved = base.joinpath(*relative.parts)
    resolved = unresolved.resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"source {source.id} escapes the catalog base directory") from exc
    return unresolved


def _inventory_file(path: Path, base: Path, source: KnowledgeSource, max_bytes: int) -> InventoryDocument:
    if path.is_symlink():
        raise ValueError(f"source {source.id} cannot inventory a symbolic link")
    if _is_blocked_vasp_artifact(path):
        raise ValueError(f"source {source.id} points to blocked VASP artifact {path.name}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"source {source.id} file exceeds max_file_bytes: {path.name} ({size})"
        )
    relative = path.resolve(strict=True).relative_to(base).as_posix()
    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(
            f"source {source.id} file changed beyond max_file_bytes: {path.name}"
        )
    return InventoryDocument(
        document_id=_document_id(source.id, relative),
        relative_path=relative,
        suffix=path.suffix.lower(),
        size_bytes=len(payload),
        sha256=digest(payload),
    )


def _directory_files(root: Path, extensions: set[str]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    warnings: list[str] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        safe_directories = []
        for name in sorted(directory_names):
            child = current / name
            if child.is_symlink():
                warnings.append(f"SYMLINK_SKIPPED:{child.name}")
            elif name in SKIP_DIRECTORY_NAMES:
                continue
            else:
                safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in sorted(file_names):
            path = current / name
            if path.is_symlink():
                warnings.append(f"SYMLINK_SKIPPED:{path.name}")
            elif _is_blocked_vasp_artifact(path):
                warnings.append(f"BLOCKED_VASP_ARTIFACT_SKIPPED:{path.name}")
            elif path.suffix.lower() in extensions:
                files.append(path)
    files.sort(key=lambda item: item.as_posix().casefold())
    return files, warnings


def _inventory_local_source(
    source: KnowledgeSource,
    base: Path,
    max_bytes: int,
    max_files: int,
) -> SourceInventory:
    path = _resolve_source(base, source)
    declaration_hash = _source_declaration_hash(source)
    location = source.path or ""
    if not path.exists():
        if source.required:
            raise FileNotFoundError(f"required knowledge source is missing: {source.id}")
        return SourceInventory(
            source_id=source.id,
            title=source.title,
            source_type=source.source_type,
            authority=source.authority,
            validated=source.validated,
            content_handling=source.content_handling,
            vasp_versions=source.vasp_versions,
            tasks=source.tasks,
            magnetic_modes=source.magnetic_modes,
            location=location,
            status=InventoryStatus.MISSING_OPTIONAL,
            declaration_sha256=declaration_hash,
            warnings=["OPTIONAL_SOURCE_MISSING"],
        )
    if path.is_symlink():
        raise ValueError(f"source {source.id} cannot be a symbolic link")

    warnings: list[str] = []
    if source.location_type == KnowledgeLocationType.LOCAL_FILE:
        if not path.is_file():
            raise ValueError(f"source {source.id} must be a file")
        files = [path]
    else:
        if not path.is_dir():
            raise ValueError(f"source {source.id} must be a directory")
        files, warnings = _directory_files(
            path, {extension.lower() for extension in source.extensions}
        )
    if len(files) > max_files:
        raise ValueError(f"source {source.id} exceeds max_files ({max_files})")
    documents = [
        _inventory_file(file_path, base, source, max_bytes) for file_path in files
    ]
    if not documents:
        warnings.append("NO_MATCHING_DOCUMENTS")
    return SourceInventory(
        source_id=source.id,
        title=source.title,
        source_type=source.source_type,
        authority=source.authority,
        validated=source.validated,
        content_handling=source.content_handling,
        vasp_versions=source.vasp_versions,
        tasks=source.tasks,
        magnetic_modes=source.magnetic_modes,
        location=location,
        status=InventoryStatus.INDEXED_METADATA,
        declaration_sha256=declaration_hash,
        documents=documents,
        warnings=warnings,
    )


def build_inventory(
    catalog: KnowledgeCatalog,
    catalog_path: Path,
    catalog_payload: bytes,
) -> KnowledgeInventory:
    base = _resolve_base(catalog_path, catalog)
    source_records: list[SourceInventory] = []
    for source in catalog.sources:
        if source.location_type == KnowledgeLocationType.EXTERNAL_URL:
            source_records.append(
                SourceInventory(
                    source_id=source.id,
                    title=source.title,
                    source_type=source.source_type,
                    authority=source.authority,
                    validated=source.validated,
                    content_handling=ContentHandling.REFERENCE_ONLY,
                    vasp_versions=source.vasp_versions,
                    tasks=source.tasks,
                    magnetic_modes=source.magnetic_modes,
                    location=str(source.url),
                    status=InventoryStatus.REFERENCE_ONLY,
                    declaration_sha256=_source_declaration_hash(source),
                    warnings=["CONTENT_NOT_CAPTURED"],
                )
            )
        else:
            source_records.append(
                _inventory_local_source(
                    source, base, catalog.max_file_bytes, catalog.max_files
                )
            )

    documents = [document for source in source_records for document in source.documents]
    by_hash: dict[str, list[str]] = {}
    for document in documents:
        by_hash.setdefault(document.sha256, []).append(document.document_id)
    duplicates = [
        DuplicateContentGroup(sha256=sha256, document_ids=sorted(document_ids))
        for sha256, document_ids in sorted(by_hash.items())
        if len(document_ids) > 1
    ]
    return KnowledgeInventory(
        schema_version=1,
        catalog_sha256=digest(catalog_payload),
        base_directory=str(base),
        sources=source_records,
        duplicate_content=duplicates,
        summary=InventorySummary(
            declared_sources=len(source_records),
            local_documents=len(documents),
            local_bytes=sum(document.size_bytes for document in documents),
            reference_only_sources=sum(
                source.status == InventoryStatus.REFERENCE_ONLY
                for source in source_records
            ),
            missing_optional_sources=sum(
                source.status == InventoryStatus.MISSING_OPTIONAL
                for source in source_records
            ),
            duplicate_content_groups=len(duplicates),
        ),
    )


def write_inventory(inventory: KnowledgeInventory, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("knowledge inventory exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(inventory.model_dump_json(indent=2))
        handle.write("\n")
