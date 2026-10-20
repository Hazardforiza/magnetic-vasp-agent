from __future__ import annotations

import hashlib
import re
from pathlib import Path

from magnetic_vasp_agent.models import PotentialRecord, PotentialRegistry

_TITEL_RE = re.compile(r"^\s*TITEL\s*=\s*(.+?)\s*$", re.MULTILINE)
_VRHFIN_RE = re.compile(r"^\s*VRHFIN\s*=\s*([^\r\n]+)", re.MULTILINE)
_LEXCH_RE = re.compile(r"^\s*LEXCH\s*=\s*([^\s;]+)", re.MULTILINE)
_ZVAL_RE = re.compile(r"\bZVAL\s*=\s*([-+0-9.Ee]+)")
_ENMAX_RE = re.compile(r"\bENMAX\s*=\s*([-+0-9.Ee]+)")
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def _match_text(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _match_float(pattern: re.Pattern[str], text: str) -> float | None:
    value = _match_text(pattern, text)
    return float(value) if value is not None else None


def _element_from_header(vrhfin: str | None, potential_name: str) -> str:
    if vrhfin:
        match = re.match(r"\s*([A-Z][a-z]?)\s*:", vrhfin)
        if match:
            return match.group(1)
    match = _ELEMENT_RE.match(potential_name)
    if not match:
        raise ValueError(f"cannot infer element from potential directory: {potential_name}")
    return match.group(1)


def scan_library(root: Path, family: str) -> PotentialRegistry:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"POTCAR library root does not exist: {root}")

    records: list[PotentialRecord] = []
    for potential_dir in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        potcar_path = potential_dir / "POTCAR"
        if not potential_dir.is_dir() or not potcar_path.is_file():
            continue

        payload = potcar_path.read_bytes()
        header = payload[:131072].decode("latin-1", errors="replace")
        titel = _match_text(_TITEL_RE, header)
        if not titel:
            raise ValueError(f"missing TITEL in {potcar_path}")
        vrhfin = _match_text(_VRHFIN_RE, header)

        records.append(
            PotentialRecord(
                element=_element_from_header(vrhfin, potential_dir.name),
                potential=potential_dir.name,
                family=family,
                titel=titel,
                relative_path=potcar_path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(payload).hexdigest(),
                vrhfin=vrhfin,
                lexch=_match_text(_LEXCH_RE, header),
                zval=_match_float(_ZVAL_RE, header),
                enmax=_match_float(_ENMAX_RE, header),
            )
        )

    if not records:
        raise ValueError(f"no POTCAR files found below {root}")

    names = [record.potential for record in records]
    if len(names) != len(set(names)):
        raise ValueError("potential directory names must be unique")

    return PotentialRegistry(family=family, root=str(root), records=records)


def write_registry(registry: PotentialRegistry, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(registry.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_registry(path: Path) -> PotentialRegistry:
    return PotentialRegistry.model_validate_json(path.read_text(encoding="utf-8"))
