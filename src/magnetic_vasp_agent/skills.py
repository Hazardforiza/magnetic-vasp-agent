"""Read-only loader for project skills. Scripts are never executed."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from magnetic_vasp_agent.skills_models import (
    ALLOWED_SKILL_SUFFIXES,
    BLOCKED_SKILL_PARTS,
    SkillCatalog,
    SkillDocument,
    SkillRecord,
    SkillSummary,
)

FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.S)
MAX_SKILL_CHARS = 200_000
DEFAULT_SKILL_CATALOG = Path("config/skills/catalog.json")


class SkillError(ValueError):
    """Raised when a skill catalog or file request is refused."""


def load_skill_catalog(path: Path) -> tuple[SkillCatalog, Path]:
    payload = path.read_bytes()
    catalog = SkillCatalog.model_validate_json(payload)
    base = _resolve_base(path, catalog)
    return catalog, base


def list_skills(catalog_path: Path) -> list[SkillSummary]:
    catalog, base = load_skill_catalog(catalog_path)
    return [_summarize(record, base) for record in catalog.skills if record.enabled]


def show_skill(catalog_path: Path, skill_id: str) -> SkillDocument:
    record, base = _find_skill(catalog_path, skill_id)
    return _read_document(record, base, "SKILL.md")


def read_skill_file(catalog_path: Path, skill_id: str, relative_path: str) -> SkillDocument:
    record, base = _find_skill(catalog_path, skill_id)
    return _read_document(record, base, relative_path)


def _resolve_base(catalog_path: Path, catalog: SkillCatalog) -> Path:
    declared = PurePosixPath(catalog.base_directory)
    if declared.is_absolute() or (declared.parts and ":" in declared.parts[0]):
        raise SkillError("skill catalog base_directory must be relative")
    candidate = (catalog_path.parent / catalog.base_directory).resolve(strict=True)
    if not candidate.is_dir():
        raise SkillError("skill catalog base_directory must resolve to a directory")
    try:
        catalog_path.resolve(strict=True).relative_to(candidate)
    except ValueError as exc:
        raise SkillError("skill catalog file must stay inside base_directory") from exc
    return candidate


def _find_skill(catalog_path: Path, skill_id: str) -> tuple[SkillRecord, Path]:
    catalog, base = load_skill_catalog(catalog_path)
    for record in catalog.skills:
        if record.id == skill_id:
            if not record.enabled:
                raise SkillError(f"skill is disabled: {skill_id}")
            return record, base
    raise SkillError(f"unknown skill: {skill_id}")


def _skill_root(record: SkillRecord, base: Path) -> Path:
    relative = PurePosixPath(record.path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SkillError(f"skill {record.id} path is unsafe")
    unresolved = base.joinpath(*relative.parts)
    resolved = unresolved.resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise SkillError(f"skill {record.id} escapes the catalog base directory") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise SkillError(f"skill {record.id} directory is missing or linked")
    return resolved


def _safe_child(root: Path, relative_path: str) -> Path:
    if "\\" in relative_path or relative_path.strip() != relative_path:
        raise SkillError("skill file path must be a trimmed POSIX-relative path")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts:
        raise SkillError("skill file path must be relative")
    if any(part in {"", ".", "..", *BLOCKED_SKILL_PARTS} for part in relative.parts):
        raise SkillError("skill file path is blocked")
    if relative.suffix.lower() not in ALLOWED_SKILL_SUFFIXES:
        raise SkillError("only Markdown or text skill files may be read")
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise SkillError("skill file escapes its skill directory") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise SkillError("skill file is missing or linked")
    return candidate


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER.match(text)
    if not match:
        return {}, text
    raw, body = match.group(1), match.group(2)
    data: dict[str, str] = {}
    current: str | None = None
    folded = False
    for line in raw.splitlines():
        if folded and current is not None and (line.startswith(" ") or line.startswith("\t")):
            data[current] = f"{data[current]} {line.strip()}".strip()
            continue
        folded = False
        if ":" not in line or line.startswith(" ") or line.startswith("\t"):
            continue
        key, _, value = line.partition(":")
        current = key.strip()
        value = value.strip()
        if value in {">", "|"}:
            data[current] = ""
            folded = True
        else:
            data[current] = value.strip("\"'")
    return data, body


def _reference_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in BLOCKED_SKILL_PARTS for part in relative.parts):
            continue
        if relative.suffix.lower() not in ALLOWED_SKILL_SUFFIXES:
            continue
        if relative.as_posix() == "SKILL.md":
            continue
        files.append(relative.as_posix())
    return files


def _summarize(record: SkillRecord, base: Path) -> SkillSummary:
    root = _skill_root(record, base)
    skill_md = _safe_child(root, "SKILL.md")
    text = skill_md.read_text(encoding="utf-8")
    meta, _body = _parse_frontmatter(text)
    return SkillSummary(
        id=record.id,
        title=record.title,
        description=str(meta.get("description") or record.notes or record.title),
        domains=list(record.domains),
        enabled=record.enabled,
        scripts_allowed=False,
        reference_files=_reference_files(root),
        notes=record.notes,
    )


def _read_document(record: SkillRecord, base: Path, relative_path: str) -> SkillDocument:
    root = _skill_root(record, base)
    path = _safe_child(root, relative_path)
    text = path.read_text(encoding="utf-8")
    if len(text) > MAX_SKILL_CHARS:
        raise SkillError(f"skill file exceeds max characters: {relative_path}")
    meta, body = _parse_frontmatter(text)
    return SkillDocument(
        id=record.id,
        title=record.title,
        description=str(meta.get("description") or record.notes or record.title),
        domains=list(record.domains),
        relative_path=path.relative_to(root).as_posix(),
        text=body if relative_path == "SKILL.md" and body.strip() else text,
        scripts_allowed=False,
    )
