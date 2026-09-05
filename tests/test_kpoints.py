from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from magnetic_vasp_agent.incar import parse_incar
from magnetic_vasp_agent.kpoints import (
    KpointsFormatError,
    parse_kpoints,
    read_kpoints,
    render_kpoints,
    validate_kpoints,
    write_kpoints,
)
from magnetic_vasp_agent.kpoints_models import KpointsSpec
from magnetic_vasp_agent.models import CalculationGoal
from magnetic_vasp_agent.poscar import read_poscar

FIXTURES = Path(__file__).parent / "fixtures"
HISTORY = json.loads((FIXTURES / "historical_kpoints.json").read_text(encoding="utf-8"))
INCARS = {
    row["source"]: row["text"]
    for row in json.loads(
        (FIXTURES / "historical_incars.json").read_text(encoding="utf-8")
    )
}


def test_historical_format_inventory() -> None:
    assert len(HISTORY) == 23
    assert Counter(parse_kpoints(row["text"]).spec.mode for row in HISTORY) == {
        "gamma": 14,
        "line": 9,
    }


@pytest.mark.parametrize("entry", HISTORY, ids=[row["source"] for row in HISTORY])
def test_all_reference_files_roundtrip_and_match_incar(entry: dict) -> None:
    doc = parse_kpoints(entry["text"])
    assert parse_kpoints(render_kpoints(doc.spec)).spec == doc.spec
    path = entry["source"]
    system = (
        "Mn3Al"
        if path.startswith("Mn3Al")
        else ("CuMnAs" if path.startswith("CuMnAs") else "SrRuO3")
    )
    incar = parse_incar(INCARS[path.removesuffix("KPOINTS") + "INCAR"]).spec
    task = "band" if "band" in path else ("dos" if "nosoc_dos" in path else "static")
    goal = CalculationGoal(
        task=task, magnetic_mode="soc" if incar.LSORBIT else "collinear"
    )
    report = validate_kpoints(
        doc.spec, read_poscar(FIXTURES / system / "POSCAR"), goal, incar
    )
    assert report.valid, report.model_dump_json()
    if incar.ISMEAR == -5:
        assert report.tetrahedron_check == "mesh_precheck_only"
        assert "TETRAHEDRON_PRECHECK" in {d.code for d in report.diagnostics}


def test_mn3al_discontinuous_path_has_six_segments_not_seven() -> None:
    entry = next(
        row for row in HISTORY if row["source"] == "Mn3Al_nosoc/nosoc_band/KPOINTS"
    )
    doc = parse_kpoints(entry["text"])
    assert doc.spec.points_per_segment == 40
    assert doc.spec.num_points == 240
    assert [(s.start.label, s.end.label) for s in doc.spec.segments] == [
        ("GAMMA", "X"),
        ("X", "U"),
        ("K", "GAMMA"),
        ("GAMMA", "L"),
        ("L", "W"),
        ("W", "X"),
    ]
    assert doc.spec.segments[1].end.coordinates == [0.625, 0.25, 0.625]
    assert doc.spec.segments[2].start.coordinates == [0.375, 0.375, 0.75]
    assert doc.source_lines["endpoint_0"] == 5


def test_synthetic_monkhorst_shift_and_explicit_fractional_weights() -> None:
    mesh = read_kpoints(FIXTURES / "kpoints_synthetic/Monkhorst.KPOINTS").spec
    assert mesh.mode == "monkhorst" and mesh.mesh == [4, 6, 8]
    assert mesh.shift == [0.5, 0, 0.5]
    assert parse_kpoints(render_kpoints(mesh)).spec == mesh
    points = read_kpoints(FIXTURES / "kpoints_synthetic/Explicit.KPOINTS").spec
    assert [p.weight for p in points.points] == [0.25] * 4
    assert [p.label for p in points.points] == ["GAMMA", "X", "Y", "Z"]
    assert points.tetrahedra.connections[0].vertices == [1, 2, 3, 4]
    assert parse_kpoints(render_kpoints(points)).spec == points


@pytest.mark.parametrize("coordinates", ["Reciprocal", "Cartesian"])
def test_path_scientific_notation_comments_and_coordinate_system(
    coordinates: str,
) -> None:
    text = f"path\r\n20 ! samples per segment\r\nLine-mode\r\n{coordinates}\r\n+0E0 0d0 -0.0 ! GAMMA\r\n+5E-1 1D-3 0 X\r\n"
    doc = parse_kpoints(text)
    assert doc.spec.coordinate_mode == coordinates.lower()
    assert doc.spec.segments[0].end.coordinates == [0.5, 0.001, 0]
    assert parse_kpoints(render_kpoints(doc.spec)).spec == doc.spec


def test_explicit_cartesian_optional_labels_and_no_tetra_block() -> None:
    text = "list\n2\nCartesian\n0 0 0 0.75\n0 0 +5D-1 0.25 ! Z\n"
    spec = parse_kpoints(text).spec
    assert spec.points[0].label is None
    assert spec.tetrahedra is None
    assert parse_kpoints(render_kpoints(spec)).spec == spec


@pytest.mark.parametrize(
    "text",
    [
        "short\n0\nGamma",
        "mesh\n0\nGamma\n2 2",
        "mesh\n0\nGamma\n2.5 2 2",
        "mesh\n2\nGamma\n2 2 2",
        "mesh\n0\nGamma\n0 2 2",
        "mesh\n0\nGamma\n2 2 2\n0 0",
        "mesh\n0\nGamma\n2 2 2\n0 0 0 0",
        "mesh\n0\nGamma\n2 2 2\n0 0 0\n1 1 1",
        "mesh\n0\nGamma\n2 2 2\n0 nan 0",
        "x\n0\nUnknown\n2 2 2",
        "x\n0\nAutomatic\n10",
        "list\n0\nReciprocal\n0 0 0",
        "list\n-1\nReciprocal\n0 0 0 1",
        "line\n1\nLine-mode\nReciprocal\n0 0 0 G\n0.5 0 0 X",
        "line\n20\nLine-mode\nUnknown\n0 0 0 G\n0.5 0 0 X",
        "line\n20\nLine-mode\nReciprocal\n0 0 0 G",
        "line\n20\nLine-mode\nReciprocal\n0 0 0 G\n0 0 0 X",
        "list\n2\nReciprocal\n0 0 0 1",
        "list\n1\nReciprocal\n0 0 0",
        "list\n1\nReciprocal\n0 0 0 -1",
        "list\n1\nReciprocal\n0 0 0 1\n0.5 0 0 1",
        "list\n1\nReciprocal\n0 0 0 1\nTetrahedron",
        "list\n1\nReciprocal\n0 0 0 1\nTetrahedron\n1 1\n1 1 2 3 4",
        "list\n4\nReciprocal\n0 0 0 1\n.5 0 0 1\n0 .5 0 1\n0 0 .5 1\nTetrahedron\n2 1\n1 1 2 3 4",
    ],
)
def test_rejects_malformed_files_instead_of_silent_truncation(text: str) -> None:
    with pytest.raises(KpointsFormatError):
        parse_kpoints(text)


@pytest.mark.parametrize(
    "data",
    [
        {"mode": "gamma", "mesh": [True, 2, 2]},
        {"mode": "gamma", "mesh": ["2", 2, 2]},
        {"mode": "gamma", "mesh": [2, 2, 2], "shift": [0, float("nan"), 0]},
        {"mode": "gamma", "mesh": [2, 2, 2], "points": []},
        {"mode": "line", "mesh": [2, 2, 2]},
        {"mode": "gamma", "mesh": [2, 2, 2], "comment": "x\n0\nGamma"},
        {
            "mode": "explicit",
            "coordinate_mode": "reciprocal",
            "points": [{"coordinates": [0, 0, 0], "weight": True}],
        },
    ],
)
def test_strict_json_contract(data: dict) -> None:
    with pytest.raises(ValidationError):
        KpointsSpec.model_validate(data)


def test_roundtrip_files_are_utf8_lf_and_do_not_overwrite(tmp_path: Path) -> None:
    spec = KpointsSpec(mode="gamma", mesh=[4, 4, 4])
    path = tmp_path / "new/KPOINTS"
    write_kpoints(spec, path)
    assert read_kpoints(path).spec == spec
    assert b"\r\n" not in path.read_bytes()
    with pytest.raises(FileExistsError):
        write_kpoints(spec, path)
