from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.incar import write_incar
from magnetic_vasp_agent.incar_models import IncarSpec
from magnetic_vasp_agent.kpoints import (
    parse_kpoints,
    read_kpoints,
    render_kpoints,
    validate_kpoints,
)
from magnetic_vasp_agent.kpoints_models import KpointsRecipeRequest, KpointsSpec
from magnetic_vasp_agent.kpoints_recipes import build_kpoints
from magnetic_vasp_agent.models import CalculationGoal, Structure

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests/fixtures"


def cell(scale: float = 2.0) -> Structure:
    return Structure(
        scale=scale,
        lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        species_order=["Fe"],
        counts=[1],
        coordinate_mode="direct",
        positions=[[0, 0, 0]],
    )


def incar(*, ismear: int = 0, band: bool = False) -> IncarSpec:
    return IncarSpec(
        ISPIN=2,
        MAGMOM=[2],
        ENCUT=600,
        ICHARG=11 if band else 2,
        ISTART=0,
        NSW=0,
        ISMEAR=ismear,
    )


def goal(task: str = "static") -> CalculationGoal:
    return CalculationGoal(task=task, magnetic_mode="collinear")


def path() -> KpointsSpec:
    return parse_kpoints("path\n40\nLine-mode\nReciprocal\n0 0 0 G\n0.5 0 0 X\n").spec


@pytest.mark.parametrize("scale", [2.0, -8.0])
def test_spacing_uses_physical_cell_and_two_pi_convention(scale: float) -> None:
    structure = cell(scale)
    before = structure.model_dump_json()
    request = KpointsRecipeRequest(goal=goal(), spacing=0.5, centering="monkhorst")
    result = build_kpoints(structure, request, incar(), incar_goal=goal())
    assert result.spec.mesh == [math.ceil(math.pi / 0.5)] * 3
    assert result.spec.mode == "monkhorst"
    assert structure.model_dump_json() == before
    assert parse_kpoints(render_kpoints(result.spec)).spec == result.spec


def test_spacing_keeps_anisotropic_cell() -> None:
    structure = cell(1.0)
    structure.lattice = [[2, 0, 0], [0, 4, 0], [0, 0, 8]]
    result = build_kpoints(
        structure,
        KpointsRecipeRequest(goal=goal(), spacing=0.5),
        incar(),
        incar_goal=goal(),
    )
    assert result.spec.mesh == [7, 4, 2]


def test_spacing_rejects_singular_cell() -> None:
    structure = cell()
    structure.lattice = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="nonsingular"):
        build_kpoints(
            structure,
            KpointsRecipeRequest(goal=goal(), spacing=0.2),
            incar(),
            incar_goal=goal(),
        )


def test_band_preserves_user_path_and_requires_matching_goal() -> None:
    request = KpointsRecipeRequest(goal=goal("band"), sampling=path())
    result = build_kpoints(cell(), request, incar(band=True), incar_goal=goal("band"))
    assert result.spec == request.sampling
    assert result.validation.incar_validation.required_artifacts == ["CHGCAR"]
    with pytest.raises(ValueError, match="goal must equal"):
        build_kpoints(cell(), request, incar(), incar_goal=goal())
    with pytest.raises(ValueError, match="TASK_CHARGE"):
        build_kpoints(cell(), request, incar(band=False), incar_goal=goal("band"))


@pytest.mark.parametrize("task", ["static", "dos", "relax", "magnetic_anisotropy"])
def test_nonband_tasks_reject_line_mode(task: str) -> None:
    task_goal = CalculationGoal(
        task=task, magnetic_mode="soc" if task == "magnetic_anisotropy" else "collinear"
    )
    report = validate_kpoints(path(), cell(), task_goal, incar())
    assert not report.valid
    assert "INTEGRATION_REQUIRES_MESH" in {d.code for d in report.diagnostics}


def test_band_rejects_mesh_and_unlabeled_endpoints() -> None:
    report = validate_kpoints(
        KpointsSpec(mode="gamma", mesh=[8, 8, 8]),
        cell(),
        goal("band"),
        incar(band=True),
    )
    assert "BAND_REQUIRES_LINE" in {d.code for d in report.diagnostics}
    unlabelled = parse_kpoints(
        "path\n40\nLine-mode\nReciprocal\n0 0 0\n.5 0 0 X\n"
    ).spec
    report = validate_kpoints(unlabelled, cell(), goal("band"), incar(band=True))
    assert "PATH_LABEL_MISSING" in {d.code for d in report.diagnostics}


def test_tetrahedron_rejects_too_few_points_and_paths() -> None:
    for mesh in ([1, 1, 1], [1, 1, 3]):
        report = validate_kpoints(
            KpointsSpec(mode="gamma", mesh=mesh), cell(), goal(), incar(ismear=-5)
        )
        assert not report.valid
        assert report.tetrahedron_check == "failed"
        assert "TETRAHEDRON_POINTS" in {d.code for d in report.diagnostics}
    report = validate_kpoints(path(), cell(), goal("band"), incar(ismear=-5, band=True))
    assert not report.valid
    assert "TETRAHEDRON_LINE" in {d.code for d in report.diagnostics}


def test_tetrahedron_auto_mesh_is_only_a_precheck() -> None:
    report = validate_kpoints(
        KpointsSpec(mode="gamma", mesh=[4, 4, 4]), cell(), goal(), incar(ismear=-5)
    )
    assert report.valid
    assert report.num_points == 64
    assert report.tetrahedron_check == "mesh_precheck_only"
    assert "TETRAHEDRON_PRECHECK" in {d.code for d in report.diagnostics}


def test_explicit_tetrahedra_required_beyond_just_point_count() -> None:
    spec = read_kpoints(FIXTURES / "kpoints_synthetic/Explicit.KPOINTS").spec
    report = validate_kpoints(spec, cell(), goal(), incar(ismear=-5))
    assert report.valid and report.tetrahedron_check == "explicit_connectivity"
    spec.tetrahedra = None
    report = validate_kpoints(spec, cell(), goal(), incar(ismear=-5))
    assert "TETRAHEDRON_MISSING" in {d.code for d in report.diagnostics}
    for point in spec.points:
        point.weight = 0.0
    report = validate_kpoints(spec, cell(), goal(), incar())
    assert "ZERO_INTEGRATION_WEIGHT" in {d.code for d in report.diagnostics}


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"spacing": "0.2"},
        {"spacing": -1.0},
        {"spacing": 0.2, "sampling": {"mode": "gamma", "mesh": [2, 2, 2]}},
        {"sampling": {"mode": "gamma", "mesh": [2, 2, 2]}, "centering": "gamma"},
    ],
)
def test_request_rejects_ambiguous_or_invalid_sampling(options: dict) -> None:
    with pytest.raises(ValidationError):
        KpointsRecipeRequest(goal=goal(), **options)
    with pytest.raises(ValidationError, match="band requires"):
        KpointsRecipeRequest(goal=goal("band"), spacing=0.2)


def test_cli_build_show_validate_manifest_and_overwrite(tmp_path: Path) -> None:
    incar_path = tmp_path / "INCAR"
    write_incar(
        IncarSpec(ISPIN=2, MAGMOM=[1.4, 1.4, -2.8, 0], ENCUT=600, ICHARG=2, ISMEAR=0),
        incar_path,
    )
    source_hash = hashlib.sha256(incar_path.read_bytes()).hexdigest()
    request_path = ROOT / "config/kpoints_requests/Mn3Al.static.json"
    incar_request = ROOT / "config/incar_requests/Mn3Al.static.json"
    poscar = FIXTURES / "Mn3Al/POSCAR"
    output = tmp_path / "new/KPOINTS"
    args = [
        "kpoints",
        "build",
        "--request",
        str(request_path),
        "--incar-request",
        str(incar_request),
        "--incar",
        str(incar_path),
        "--poscar",
        str(poscar),
        "--output",
        str(output),
    ]
    runner = CliRunner()
    built = runner.invoke(app, args)
    assert built.exit_code == 0, built.output
    manifest = json.loads(output.with_name("KPOINTS.manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["goal"]["task"] == "static"
    assert manifest["validation"]["valid"]
    assert manifest["sources"]["incar"]["sha256"] == source_hash
    assert read_kpoints(output).spec.mesh == [18, 18, 18]
    shown = runner.invoke(app, ["kpoints", "show", str(output)])
    assert shown.exit_code == 0 and json.loads(shown.output)["spec"]["mode"] == "gamma"
    validate_args = [
        "kpoints",
        "validate",
        str(output),
        "--incar-request",
        str(incar_request),
        "--incar",
        str(incar_path),
        "--poscar",
        str(poscar),
    ]
    checked = runner.invoke(app, validate_args)
    assert checked.exit_code == 0, checked.output
    assert runner.invoke(app, args).exit_code == 2
    assert hashlib.sha256(output.read_bytes()).hexdigest() == manifest["sha256"]
    assert hashlib.sha256(incar_path.read_bytes()).hexdigest() == source_hash


def test_cli_rejects_goal_conflict_before_writing(tmp_path: Path) -> None:
    incar_path = tmp_path / "INCAR"
    write_incar(IncarSpec(ISPIN=2, MAGMOM=[1.4, 1.4, -2.8, 0], ICHARG=2), incar_path)
    output = tmp_path / "KPOINTS"
    result = CliRunner().invoke(
        app,
        [
            "kpoints",
            "build",
            "--request",
            str(ROOT / "config/kpoints_requests/Mn3Al.band.json"),
            "--incar-request",
            str(ROOT / "config/incar_requests/Mn3Al.static.json"),
            "--incar",
            str(incar_path),
            "--poscar",
            str(FIXTURES / "Mn3Al/POSCAR"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2 and "goal must equal" in result.output
    assert not output.exists()
