"""Strict parsers and static plots for selected VASPKIT text outputs."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

_MAX_DATA_BYTES = 128 * 1024 * 1024
_DOS_HEADER = re.compile(
    r"^#\s*Energy(?:\s*\([^)]*\))?\s+TDOS\s*$", re.IGNORECASE
)
_SPIN_DOS_HEADER = re.compile(
    r"^#\s*Energy(?:\s*\([^)]*\))?\s+TDOS-UP\s+TDOS-DOWN\s*$",
    re.IGNORECASE,
)
_BAND_INDEX = re.compile(r"^#\s*Band[- ]index\s*:?[ \t]*(\d+)\s*$", re.IGNORECASE)


class FileHash(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    size_bytes: int


class VaspkitPlotResult(BaseModel):
    """Auditable metadata for a generated, static PNG."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    plot_kind: Literal["dos", "band"]
    data_format: str
    data: FileHash
    labels: FileHash | None = None
    points: int
    series: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    plot: FileHash


@dataclass(frozen=True)
class DosTable:
    energies: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    labels: tuple[str, ...]
    data_format: str


@dataclass(frozen=True)
class BandTable:
    kpath: tuple[float, ...]
    bands: tuple[tuple[float, ...], ...]
    data_format: str


@dataclass(frozen=True)
class KLabels:
    labels: tuple[str, ...]
    coordinates: tuple[float, ...]


def _regular_file(path: Path, description: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{description} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{description} must be a regular, non-symlink file: {path}")
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError(f"{description} is empty: {resolved}")
    if size > _MAX_DATA_BYTES:
        raise ValueError(f"{description} exceeds {_MAX_DATA_BYTES} bytes: {resolved}")
    return resolved


def _read_ascii_table(path: Path, description: str) -> tuple[Path, list[str]]:
    resolved = _regular_file(path, description)
    try:
        text = resolved.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{description} must be ASCII text: {resolved}") from exc
    if "\x00" in text:
        raise ValueError(f"{description} contains NUL bytes: {resolved}")
    return resolved, text.splitlines()


def _numbers(line: str, *, expected: int | None, description: str) -> tuple[float, ...]:
    fields = line.split()
    if expected is not None and len(fields) != expected:
        raise ValueError(
            f"{description} has {len(fields)} columns; expected exactly {expected}: {line!r}"
        )
    try:
        values = tuple(float(field) for field in fields)
    except ValueError as exc:
        raise ValueError(f"{description} contains a non-numeric row: {line!r}") from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{description} contains a non-finite or empty row: {line!r}")
    return values


def _require_monotonic(
    values: tuple[float, ...], *, strict: bool, description: str
) -> None:
    if len(values) < 2:
        raise ValueError(f"{description} requires at least two data points")
    pairs = zip(values, values[1:], strict=False)
    valid = all(right > left if strict else right >= left for left, right in pairs)
    if not valid:
        qualifier = "strictly increasing" if strict else "non-decreasing"
        raise ValueError(f"{description} coordinates must be {qualifier}")


def parse_tdos(path: Path) -> DosTable:
    """Parse only the documented VASPKIT total-DOS column layouts."""

    _, lines = _read_ascii_table(path, "TDOS data")
    header: str | None = None
    rows: list[tuple[float, ...]] = []
    expected_columns: int | None = None
    labels: tuple[str, ...] = ()
    data_format = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if header is None:
                header = line
                if _DOS_HEADER.fullmatch(line):
                    expected_columns = 2
                    labels = ("TDOS",)
                    data_format = "vaspkit_tdos_nonspin_v1"
                elif _SPIN_DOS_HEADER.fullmatch(line):
                    expected_columns = 3
                    labels = ("TDOS-UP", "TDOS-DOWN")
                    data_format = "vaspkit_tdos_spin_v1"
                else:
                    raise ValueError(f"unsupported TDOS header: {line!r}")
            continue
        if expected_columns is None:
            raise ValueError("TDOS data must start with a supported #Energy header")
        rows.append(_numbers(line, expected=expected_columns, description="TDOS data"))

    if header is None or expected_columns is None:
        raise ValueError("TDOS data has no supported header")
    if len(rows) < 2:
        raise ValueError("TDOS data requires at least two numeric rows")
    energies = tuple(row[0] for row in rows)
    _require_monotonic(energies, strict=True, description="TDOS energy")
    series = tuple(tuple(row[index] for row in rows) for index in range(1, expected_columns))
    return DosTable(energies, series, labels, data_format)


def _parse_band_blocks(lines: list[str]) -> BandTable | None:
    blocks: list[tuple[int, list[tuple[float, ...]]]] = []
    current_index: int | None = None
    current_rows: list[tuple[float, ...]] = []
    saw_kpath_header = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "K-Path" in line:
                saw_kpath_header = True
            match = _BAND_INDEX.fullmatch(line)
            if match:
                if current_index is not None:
                    blocks.append((current_index, current_rows))
                current_index = int(match.group(1))
                current_rows = []
            continue
        if current_index is None:
            continue
        row = _numbers(line, expected=None, description="BAND data")
        if len(row) not in {2, 3}:
            raise ValueError("BAND block rows must have exactly 2 or 3 columns")
        current_rows.append(row)

    if current_index is not None:
        blocks.append((current_index, current_rows))
    if not blocks:
        return None
    if not saw_kpath_header:
        raise ValueError("BAND data is missing a #K-Path header")
    if len({index for index, _ in blocks}) != len(blocks):
        raise ValueError("BAND data repeats a band index")
    if any(not rows for _, rows in blocks):
        raise ValueError("BAND data contains an empty band block")

    reference_x = tuple(row[0] for row in blocks[0][1])
    _require_monotonic(reference_x, strict=False, description="BAND k-path")
    width = len(blocks[0][1][0])
    bands: list[tuple[float, ...]] = []
    for _, rows in blocks:
        if len(rows) != len(reference_x) or any(len(row) != width for row in rows):
            raise ValueError("BAND blocks have inconsistent shapes")
        x_values = tuple(row[0] for row in rows)
        if any(abs(left - right) > 1e-8 for left, right in zip(x_values, reference_x, strict=True)):
            raise ValueError("BAND blocks do not share the same k-path coordinates")
        bands.extend(tuple(row[column] for row in rows) for column in range(1, width))
    return BandTable(reference_x, tuple(bands), "vaspkit_band_blocks_v1")


def _parse_reformatted_band(lines: list[str]) -> BandTable:
    header = next((line.strip() for line in lines if line.strip()), "")
    if not header.startswith("#") or "K-Path" not in header:
        raise ValueError("reformatted BAND data must start with a #K-Path header")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        row = _numbers(line, expected=width, description="reformatted BAND data")
        if len(row) < 2:
            raise ValueError("reformatted BAND data needs k-path plus at least one band")
        width = len(row)
        rows.append(row)
    if len(rows) < 2 or width is None:
        raise ValueError("reformatted BAND data requires at least two numeric rows")
    kpath = tuple(row[0] for row in rows)
    _require_monotonic(kpath, strict=False, description="BAND k-path")
    if len(set(kpath)) < 2:
        raise ValueError("BAND k-path requires at least two distinct coordinates")
    bands = tuple(tuple(row[index] for row in rows) for index in range(1, width))
    return BandTable(kpath, bands, "vaspkit_reformatted_band_v1")


def parse_band(path: Path) -> BandTable:
    """Parse VASPKIT BAND.dat blocks or REFORMATTED_BAND.dat matrices."""

    _, lines = _read_ascii_table(path, "band data")
    blocked = _parse_band_blocks(lines)
    return blocked if blocked is not None else _parse_reformatted_band(lines)


def parse_klabels(path: Path, kpath: tuple[float, ...]) -> KLabels:
    """Parse KLABELS and require its coordinates to match the band domain."""

    _, lines = _read_ascii_table(path, "KLABELS")
    first = next((line.strip() for line in lines if line.strip()), "")
    if not first.lower().startswith("k-label"):
        raise ValueError("KLABELS must start with a K-Label header")
    labels: list[str] = []
    coordinates: list[float] = []
    for raw in lines[1:]:
        line = raw.strip()
        if not line or line.startswith("*") or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"KLABELS row must have exactly two fields: {line!r}")
        try:
            coordinate = float(fields[1])
        except ValueError as exc:
            raise ValueError(f"KLABELS coordinate is not numeric: {line!r}") from exc
        if not math.isfinite(coordinate):
            raise ValueError(f"KLABELS coordinate is not finite: {line!r}")
        labels.append(fields[0])
        coordinates.append(coordinate)
    coordinate_tuple = tuple(coordinates)
    _require_monotonic(coordinate_tuple, strict=False, description="KLABELS")
    tolerance = max(1e-6, (kpath[-1] - kpath[0]) * 1e-4)
    if abs(coordinate_tuple[0] - kpath[0]) > tolerance:
        raise ValueError("first KLABELS coordinate does not match the band k-path")
    if abs(coordinate_tuple[-1] - kpath[-1]) > tolerance:
        raise ValueError("last KLABELS coordinate does not match the band k-path")
    if any(value < kpath[0] - tolerance or value > kpath[-1] + tolerance for value in coordinate_tuple):
        raise ValueError("KLABELS coordinate lies outside the band k-path")
    return KLabels(tuple(labels), coordinate_tuple)


def _digest(path: Path) -> FileHash:
    payload = path.read_bytes()
    return FileHash(
        path=str(path.resolve()),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _new_png(path: Path) -> Path:
    if path.suffix.lower() != ".png":
        raise ValueError("plot output must use a .png suffix")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"plot output exists; choose a new path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def plot_tdos(data_path: Path, output: Path) -> VaspkitPlotResult:
    """Render a static total-DOS plot from a strictly parsed TDOS.dat."""

    source = _regular_file(data_path, "TDOS data")
    table = parse_tdos(source)
    target = _new_png(output)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = ("#b42318", "#175cd3")
    for index, (label, values) in enumerate(zip(table.labels, table.values, strict=True)):
        axis.plot(table.energies, values, label=label, color=colors[index], linewidth=1.35)
    axis.axvline(0.0, color="#667085", linewidth=0.8, linestyle="--")
    axis.set_xlabel(r"Energy - $E_F$ (eV)")
    axis.set_ylabel("Density of states")
    axis.grid(axis="y", color="#d0d5dd", linewidth=0.55, alpha=0.7)
    axis.legend(frameon=False)
    figure.savefig(
        target,
        format="png",
        dpi=180,
        metadata={"Software": "magnetic-vasp-agent"},
    )
    plt.close(figure)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("matplotlib did not produce a non-empty DOS PNG")
    all_y = tuple(value for series in table.values for value in series)
    return VaspkitPlotResult(
        plot_kind="dos",
        data_format=table.data_format,
        data=_digest(source),
        points=len(table.energies),
        series=len(table.values),
        x_range=(min(table.energies), max(table.energies)),
        y_range=(min(all_y), max(all_y)),
        plot=_digest(target),
    )


def plot_band(data_path: Path, labels_path: Path, output: Path) -> VaspkitPlotResult:
    """Render a static band plot from VASPKIT band data and KLABELS."""

    source = _regular_file(data_path, "band data")
    label_source = _regular_file(labels_path, "KLABELS")
    table = parse_band(source)
    labels = parse_klabels(label_source, table.kpath)
    target = _new_png(output)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for values in table.bands:
        axis.plot(table.kpath, values, color="#175cd3", linewidth=0.8)
    axis.axhline(0.0, color="#667085", linewidth=0.8, linestyle="--")
    for coordinate in labels.coordinates[1:-1]:
        axis.axvline(coordinate, color="#98a2b3", linewidth=0.55)
    display_labels = [r"$\Gamma$" if label.upper() == "GAMMA" else label for label in labels.labels]
    axis.set_xticks(labels.coordinates, display_labels)
    axis.set_xlim(table.kpath[0], table.kpath[-1])
    axis.set_xlabel("Wave vector")
    axis.set_ylabel(r"Energy - $E_F$ (eV)")
    axis.grid(axis="y", color="#d0d5dd", linewidth=0.5, alpha=0.6)
    figure.savefig(
        target,
        format="png",
        dpi=180,
        metadata={"Software": "magnetic-vasp-agent"},
    )
    plt.close(figure)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("matplotlib did not produce a non-empty band PNG")
    all_y = tuple(value for band in table.bands for value in band)
    return VaspkitPlotResult(
        plot_kind="band",
        data_format=table.data_format,
        data=_digest(source),
        labels=_digest(label_source),
        points=len(table.kpath),
        series=len(table.bands),
        x_range=(min(table.kpath), max(table.kpath)),
        y_range=(min(all_y), max(all_y)),
        plot=_digest(target),
    )
