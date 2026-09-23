"""Strict KPOINTS adapter around pymatgen, including lossless real weights."""

from __future__ import annotations

import math
import re
from decimal import Decimal
from pathlib import Path

from pymatgen.io.vasp.inputs import Kpoints

from magnetic_vasp_agent.incar import validate_incar
from magnetic_vasp_agent.incar_models import IncarSpec
from magnetic_vasp_agent.kpoints_models import (
    KpointsDiagnostic,
    KpointsSpec,
    KpointsValidation,
    LineSegment,
    ParsedKpoints,
    PathPoint,
    Tetrahedra,
    Tetrahedron,
    WeightedPoint,
)
from magnetic_vasp_agent.models import CalculationGoal, Structure


class KpointsFormatError(ValueError):
    pass


def _body(line: str) -> str:
    return re.split(r"[!#]", line, maxsplit=1)[0].strip()


def _integer(token: str, line: int) -> int:
    if not re.fullmatch(r"[+-]?\d+", token):
        raise KpointsFormatError(f"line {line}: expected integer, got {token!r}")
    return int(token)


def _real(token: str, line: int) -> float:
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?", token):
        raise KpointsFormatError(f"line {line}: invalid number {token!r}")
    normalized = token.replace("D", "E").replace("d", "e")
    value = float(normalized)
    if not math.isfinite(value) or (value == 0 and Decimal(normalized) != 0):
        raise KpointsFormatError(f"line {line}: number is outside finite float range")
    return value


def _fixed(values) -> str:
    # pymatgen's Line-mode regex misses uppercase E, D and leading '+' tokens.
    return " ".join(format(Decimal(str(value)), "f") for value in values)


def _row(line: str, number: int, columns: int, *, label_allowed: bool = False):
    fields = line.split(maxsplit=columns)
    if len(fields) < columns:
        raise KpointsFormatError(f"line {number}: expected {columns} numbers")
    values = [_real(token, number) for token in fields[:columns]]
    label = None
    if len(fields) > columns:
        tail = fields[-1].strip()
        if label_allowed:
            label = tail.lstrip("!#").strip() or None
        elif not tail.startswith(("!", "#")):
            raise KpointsFormatError(f"line {number}: unexpected trailing values")
    return values, label


def _coordinate_mode(line: str, number: int) -> str:
    mode = _body(line).lower()
    if mode in {"r", "reciprocal"}:
        return "reciprocal"
    if mode in {"c", "k", "cartesian", "cart"}:
        return "cartesian"
    raise KpointsFormatError(f"line {number}: expected Reciprocal or Cartesian")


def parse_kpoints(text: str) -> ParsedKpoints:
    lines = text.lstrip("\ufeff").splitlines()
    if len(lines) < 4:
        raise KpointsFormatError("KPOINTS needs at least four lines")
    count = _integer(_body(lines[1]), 2)
    if not 0 <= count <= 1000000:
        raise KpointsFormatError("line 2: count must be between 0 and 1000000")
    mode_text = _body(lines[2]).lower()
    data = [
        (i + 1, line.strip())
        for i, line in enumerate(lines[3:], 3)
        if line.strip() and not line.lstrip().startswith(("!", "#"))
    ]
    source_lines = {"count": 2, "mode": 3}
    try:
        if mode_text in {
            "g",
            "gamma",
            "m",
            "monkhorst",
            "monkhorst-pack",
            "monkhorst_pack",
        }:
            if count != 0:
                raise KpointsFormatError("line 2: automatic mesh requires count=0")
            if len(data) not in {1, 2}:
                raise KpointsFormatError(
                    "automatic mesh needs one mesh row and at most one shift row"
                )
            tokens = _body(data[0][1]).split()
            if len(tokens) != 3:
                raise KpointsFormatError(
                    f"line {data[0][0]}: mesh requires three integers"
                )
            mesh = [_integer(token, data[0][0]) for token in tokens]
            shift = (
                _row(data[1][1], data[1][0], 3)[0]
                if len(data) == 2
                else [0.0, 0.0, 0.0]
            )
            mode = "gamma" if mode_text.startswith("g") else "monkhorst"
            canonical = f"{lines[0]}\n0\n{'Gamma' if mode == 'gamma' else 'Monkhorst-Pack'}\n{' '.join(map(str, mesh))}\n{_fixed(shift)}\n"
            parsed = Kpoints.from_str(canonical)
            spec = KpointsSpec(
                mode=mode,
                comment=parsed.comment,
                mesh=list(parsed.kpts[0]),
                shift=list(parsed.kpts_shift),
            )
            source_lines["mesh"] = data[0][0]
            if len(data) == 2:
                source_lines["shift"] = data[1][0]
        elif mode_text in {"l", "line", "line-mode", "line_mode"}:
            if count < 2:
                raise KpointsFormatError(
                    "line 2: Line-mode needs at least two points per segment"
                )
            if not data:
                raise KpointsFormatError("missing Line-mode coordinate system")
            coordinate_mode = _coordinate_mode(data[0][1], data[0][0])
            endpoints = data[1:]
            if len(endpoints) < 2 or len(endpoints) % 2:
                raise KpointsFormatError(
                    "Line-mode endpoints must form complete start/end pairs"
                )
            rows = [
                _row(line, number, 3, label_allowed=True) for number, line in endpoints
            ]
            canonical = (
                f"{lines[0]}\n{count}\nLine-mode\n{coordinate_mode.capitalize()}\n"
            )
            canonical += (
                "\n".join(
                    _fixed(values) + (f" ! {label}" if label else "")
                    for values, label in rows
                )
                + "\n"
            )
            parsed = Kpoints.from_str(canonical)
            if len(parsed.kpts) != len(rows):
                raise KpointsFormatError("pymatgen did not preserve the endpoint count")
            points = [
                PathPoint(coordinates=list(kpt), label=label)
                for kpt, (_, label) in zip(parsed.kpts, rows, strict=True)
            ]
            spec = KpointsSpec(
                mode="line",
                comment=parsed.comment,
                coordinate_mode=coordinate_mode,
                points_per_segment=count,
                segments=[
                    LineSegment(start=points[i], end=points[i + 1])
                    for i in range(0, len(points), 2)
                ],
            )
            source_lines.update(
                {f"endpoint_{i}": number for i, (number, _) in enumerate(endpoints)}
            )
        else:
            coordinate_mode = _coordinate_mode(lines[2], 3)
            if count < 1:
                raise KpointsFormatError(
                    "explicit lists require positive count; automatic basis mode is unsupported"
                )
            if len(data) < count:
                raise KpointsFormatError(
                    f"expected {count} explicit points, found {len(data)}"
                )
            rows = [
                _row(line, number, 4, label_allowed=True)
                for number, line in data[:count]
            ]
            tail = data[count:]
            tetrahedra = None
            tetra_text = ""
            if tail:
                if (
                    _body(tail[0][1]).lower() not in {"t", "tetrahedron", "tetrahedra"}
                    or len(tail) < 3
                ):
                    raise KpointsFormatError(
                        f"line {tail[0][0]}: unexpected or incomplete tetrahedron block"
                    )
                header = _body(tail[1][1]).split()
                if len(header) != 2:
                    raise KpointsFormatError(
                        f"line {tail[1][0]}: expected tetrahedron count and volume"
                    )
                tet_count = _integer(header[0], tail[1][0])
                volume = _real(header[1], tail[1][0])
                if tet_count < 1 or len(tail) != tet_count + 2:
                    raise KpointsFormatError(
                        "tetrahedron count does not match connectivity rows"
                    )
                connections = []
                for number, line in tail[2:]:
                    tokens = _body(line).split()
                    if len(tokens) != 5:
                        raise KpointsFormatError(
                            f"line {number}: expected multiplicity and four indices"
                        )
                    values = [_integer(token, number) for token in tokens]
                    connections.append(
                        Tetrahedron(multiplicity=values[0], vertices=values[1:])
                    )
                tetrahedra = Tetrahedra(volume=volume, connections=connections)
                tetra_text = f"Tetrahedron\n{tet_count} {_fixed([volume])}\n"
                tetra_text += (
                    "\n".join(
                        " ".join(map(str, [tet.multiplicity, *tet.vertices]))
                        for tet in connections
                    )
                    + "\n"
                )
            canonical = f"{lines[0]}\n{count}\n{coordinate_mode.capitalize()}\n"
            canonical += (
                "\n".join(_fixed(values) for values, _ in rows) + "\n" + tetra_text
            )
            parsed = Kpoints.from_str(canonical)
            spec = KpointsSpec(
                mode="explicit",
                comment=parsed.comment,
                coordinate_mode=coordinate_mode,
                points=[
                    WeightedPoint(coordinates=list(kpt), weight=weight, label=label)
                    for kpt, weight, (_, label) in zip(
                        parsed.kpts, parsed.kpts_weights, rows, strict=True
                    )
                ],
                tetrahedra=tetrahedra,
            )
            source_lines.update(
                {f"point_{i}": number for i, (number, _) in enumerate(data[:count])}
            )
    except (ValueError, IndexError, TypeError, OverflowError) as exc:
        if isinstance(exc, KpointsFormatError):
            raise
        raise KpointsFormatError(f"KPOINTS {mode_text!r}: {exc}") from exc
    return ParsedKpoints(spec=spec, source_lines=source_lines)


def read_kpoints(path: Path) -> ParsedKpoints:
    return parse_kpoints(path.read_text(encoding="utf-8-sig"))


def render_kpoints(spec: KpointsSpec) -> str:
    spec = KpointsSpec.model_validate(spec.model_dump())
    if spec.mode in {"gamma", "monkhorst"}:
        factory = (
            Kpoints.gamma_automatic
            if spec.mode == "gamma"
            else Kpoints.monkhorst_automatic
        )
        return str(factory(tuple(spec.mesh), tuple(spec.shift), comment=spec.comment))
    if spec.mode == "line":
        endpoints = [
            point for segment in spec.segments for point in (segment.start, segment.end)
        ]
        return str(
            Kpoints(
                comment=spec.comment,
                num_kpts=spec.points_per_segment,
                style=Kpoints.supported_modes.Line_mode,
                coord_type=spec.coordinate_mode.capitalize(),
                kpts=[point.coordinates for point in endpoints],
                labels=[point.label or "" for point in endpoints],
            )
        )
    # pymatgen's writer casts explicit weights to int; retain fractional weights.
    lines = [spec.comment, str(len(spec.points)), spec.coordinate_mode.capitalize()]
    for point in spec.points:
        lines.append(
            _fixed([*point.coordinates, point.weight])
            + (f" ! {point.label}" if point.label else "")
        )
    if spec.tetrahedra is not None:
        lines += [
            "Tetrahedron",
            f"{len(spec.tetrahedra.connections)} {_fixed([spec.tetrahedra.volume])}",
        ]
        lines += [
            " ".join(map(str, [tet.multiplicity, *tet.vertices]))
            for tet in spec.tetrahedra.connections
        ]
    return "\n".join(lines) + "\n"


def write_kpoints(spec: KpointsSpec, path: Path) -> None:
    content = render_kpoints(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def validate_kpoints(
    spec: KpointsSpec, structure: Structure, goal: CalculationGoal, incar: IncarSpec
) -> KpointsValidation:
    spec = KpointsSpec.model_validate(spec.model_dump())
    report = KpointsValidation(
        task=goal.task.value,
        num_points=spec.num_points,
        incar_validation=validate_incar(incar, structure, goal=goal),
    )

    def issue(code: str, message: str, severity: str = "error") -> None:
        report.diagnostics.append(
            KpointsDiagnostic(code=code, message=message, severity=severity)
        )

    if goal.task.value == "band" and spec.mode != "line":
        issue(
            "BAND_REQUIRES_LINE",
            "Project band tasks require Line-mode; a weighted list is not a band path in this workflow.",
        )
    if goal.task.value != "band" and spec.mode == "line":
        issue(
            "INTEGRATION_REQUIRES_MESH",
            "SCF, DOS, relaxation and MAE require integration sampling, not Line-mode.",
        )
    if spec.mode == "line":
        if any(
            point.label is None
            for segment in spec.segments
            for point in (segment.start, segment.end)
        ):
            issue(
                "PATH_LABEL_MISSING",
                "Every segment endpoint needs a label for a traceable band path.",
            )
    if spec.mode == "explicit" and not any(point.weight > 0 for point in spec.points):
        issue(
            "ZERO_INTEGRATION_WEIGHT",
            "An explicit integration list needs positive total weight.",
        )
    if incar.ISMEAR in {-5, -4}:
        if spec.mode == "line":
            issue(
                "TETRAHEDRON_LINE",
                "Tetrahedron integration cannot use a Line-mode path.",
            )
            report.tetrahedron_check = "failed"
        elif spec.num_points < 4:
            issue(
                "TETRAHEDRON_POINTS",
                "Tetrahedron integration needs at least four k points; this sampling has fewer.",
            )
            report.tetrahedron_check = "failed"
        elif spec.mode in {"gamma", "monkhorst"}:
            report.tetrahedron_check = "mesh_precheck_only"
            issue(
                "TETRAHEDRON_PRECHECK",
                "Full mesh has at least four points. Actual irreducible points and tetrahedron construction must be verified from VASP/IBZKPT; this is only a necessary precheck.",
                "warning",
            )
        elif spec.tetrahedra is None:
            issue(
                "TETRAHEDRON_MISSING",
                "Explicit points need an explicit tetrahedron connectivity block for ISMEAR=-5/-4.",
            )
            report.tetrahedron_check = "failed"
        else:
            report.tetrahedron_check = "explicit_connectivity"
            issue(
                "TETRAHEDRON_CONNECTIVITY",
                "Connectivity indices and volume passed structural checks; Brillouin-zone tessellation and integration convergence are not certified.",
                "warning",
            )
    return report
