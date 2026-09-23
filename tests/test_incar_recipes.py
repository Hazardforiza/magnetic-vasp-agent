from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.incar import parse_incar, render_incar, validate_incar
from magnetic_vasp_agent.incar_models import IncarRecipeRequest, IncarSpec
from magnetic_vasp_agent.incar_recipes import build_incar
from magnetic_vasp_agent.models import (
    CalculationGoal,
    CalculationTask,
    CandidateComposition,
    MagneticMode,
    PotentialPolicy,
    PotentialRecord,
    PotentialRegistry,
)
from magnetic_vasp_agent.poscar import read_poscar
from magnetic_vasp_agent.potcar.selector import select_by_policy

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests/fixtures"


def context(system: str, goal: CalculationGoal):
    structure = read_poscar(FIXTURES / system / "POSCAR")
    registry = PotentialRegistry(
        family="PBE_54",
        root="unused-test-library",
        records=[
            PotentialRecord(
                element=e,
                potential=e,
                family="PBE_54",
                titel=f"PAW_PBE {e}",
                relative_path=f"{e}/POTCAR",
                sha256="a" * 64,
                enmax=400.0,
            )
            for e in structure.species_order
        ],
    )
    plan = select_by_policy(
        CandidateComposition.from_structure(structure),
        goal,
        registry,
        PotentialPolicy(family="PBE_54", preferred={}),
    )
    return structure, plan, registry


@pytest.mark.parametrize("task", list(CalculationTask))
@pytest.mark.parametrize("mode", list(MagneticMode))
def test_task_and_spin_mode_recipes(task, mode) -> None:
    if task == CalculationTask.MAGNETIC_ANISOTROPY and mode != MagneticMode.SOC:
        with pytest.raises(ValidationError, match="requires magnetic_mode='soc'"):
            CalculationGoal(task=task, magnetic_mode=mode)
        return
    goal = CalculationGoal(task=task, magnetic_mode=mode)
    structure, plan, registry = context("Mn3Al", goal)
    moments = (
        [1.4, 1.4, -2.8, 0]
        if mode == MagneticMode.COLLINEAR
        else [0, 0, 1.4, 0, 0, 1.4, 0, 0, -2.8, 0, 0, 0]
    )
    result = build_incar(
        structure,
        IncarRecipeRequest(goal=goal, encut=600, magmom=moments),
        plan,
        registry,
    )
    assert result.validation.valid
    assert result.spec.MAGMOM == moments
    assert parse_incar(render_incar(result.spec)).spec == result.spec
    assert result.validation.executable == (
        "vasp_std" if mode == MagneticMode.COLLINEAR else "vasp_ncl"
    )
    if mode != MagneticMode.COLLINEAR:
        assert result.spec.ISPIN is None
        assert result.spec.LNONCOLLINEAR is True
    if task in {CalculationTask.DOS, CalculationTask.BAND}:
        assert result.validation.required_artifacts == ["CHGCAR"]
        assert result.spec.ICHARG == 11
        assert result.spec.LCHARG is False
    else:
        assert not result.validation.required_artifacts
    assert result.spec.NSW == (200 if task == CalculationTask.RELAX else 0)


@pytest.mark.parametrize(
    "system,request_file",
    [
        ("Mn3Al", "Mn3Al.static.json"),
        ("CuMnAs", "CuMnAs.soc.json"),
        ("SrRuO3", "SrRuO3.soc-u.json"),
    ],
)
def test_checked_in_requests(system: str, request_file: str) -> None:
    request = IncarRecipeRequest.model_validate_json(
        (ROOT / "config/incar_requests" / request_file).read_text()
    )
    structure, plan, registry = context(system, request.goal)
    result = build_incar(structure, request, plan, registry)
    assert result.validation.valid
    if system == "CuMnAs":
        assert result.spec.MAGMOM[-6:] == [0, 0, 6, 0, 0, -6]
    if system == "SrRuO3":
        assert result.spec.SAXIS == [1, 0, 0]
        assert result.spec.LDAUL == [-1, 2, -1]
        assert result.spec.LDAUU == [0, 1.2, 0]
        assert result.spec.LDAUJ == [0, 0.6, 0]


def test_transverse_moments_and_spin_basis_are_preserved() -> None:
    goal = CalculationGoal(task="static", magnetic_mode="noncollinear")
    structure, plan, registry = context("Mn3Al", goal)
    moments = [1, 2, 3, -1, -2, -3, 0.1, -0.2, 0.3, 0, 0, 0]
    result = build_incar(
        structure,
        IncarRecipeRequest(
            goal=goal,
            encut=600,
            magmom=moments,
            saxis=[1, 1, 1],
        ),
        plan,
        registry,
    )
    spec = parse_incar(render_incar(result.spec)).spec
    assert spec.MAGMOM == moments
    assert spec.SAXIS == [1, 1, 1]
    assert spec.LNONCOLLINEAR is True and spec.LSORBIT is False


def test_relax_goal_rejects_static_input() -> None:
    goal = CalculationGoal(task="relax", magnetic_mode="collinear")
    structure, _, _ = context("Mn3Al", goal)
    report = validate_incar(
        IncarSpec(ISPIN=2, MAGMOM=[1, 1, -2, 0], NSW=0), structure, goal=goal
    )
    assert "RELAX_STEPS" in {d.code for d in report.diagnostics}


def test_selected_enmax_not_entire_library_and_missing_context() -> None:
    goal = CalculationGoal(task="static", magnetic_mode="collinear")
    structure, plan, registry = context("Mn3Al", goal)
    registry.records.append(
        PotentialRecord(
            element="O",
            potential="O",
            family="PBE_54",
            titel="PAW_PBE O",
            relative_path="O/POTCAR",
            sha256="b" * 64,
            enmax=1000,
        )
    )
    good = IncarSpec(ISPIN=2, MAGMOM=[1, 1, -2, 0], ENCUT=400)
    assert validate_incar(good, structure, plan=plan, registry=registry).valid
    low = IncarSpec(ISPIN=2, MAGMOM=[1, 1, -2, 0], ENCUT=399)
    assert not validate_incar(low, structure, plan=plan, registry=registry).valid
    assert not validate_incar(good, structure, plan=plan).valid
    plan.species_order = list(reversed(plan.species_order))
    assert not validate_incar(good, structure, plan=plan, registry=registry).valid


def test_build_rejects_wrong_moments_unknown_u_species_and_goal() -> None:
    goal = CalculationGoal(task="static", magnetic_mode="soc")
    structure, plan, registry = context("Mn3Al", goal)
    request = IncarRecipeRequest(goal=goal, encut=600, magmom=[1, 1, -2, 0])
    with pytest.raises(ValueError, match="MAGMOM_LENGTH"):
        build_incar(structure, request, plan, registry)
    request = IncarRecipeRequest(
        goal=goal, encut=600, magmom=[0, 0, 1] * 4, hubbard={"Fe": {"l": 2, "u": 3.0}}
    )
    with pytest.raises(ValueError, match="absent from POSCAR"):
        build_incar(structure, request, plan, registry)
    request.goal = CalculationGoal(task="dos", magnetic_mode="soc")
    with pytest.raises(ValueError, match="goal"):
        build_incar(structure, request, plan, registry)


def test_cli_build_manifest_and_refuse_overwrite(tmp_path: Path) -> None:
    request_path = ROOT / "config/incar_requests/Mn3Al.static.json"
    request = IncarRecipeRequest.model_validate_json(request_path.read_text())
    _, plan, registry = context("Mn3Al", request.goal)
    for name, model in [("plan", plan), ("registry", registry)]:
        (tmp_path / f"{name}.json").write_text(
            model.model_dump_json(), encoding="utf-8"
        )
    output = tmp_path / "new/INCAR"
    args = [
        "incar",
        "build",
        "--request",
        str(request_path),
        "--poscar",
        str(FIXTURES / "Mn3Al/POSCAR"),
        "--plan",
        str(tmp_path / "plan.json"),
        "--registry",
        str(tmp_path / "registry.json"),
        "--output",
        str(output),
    ]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    manifest = json.loads(output.with_name("INCAR.manifest.json").read_text())
    assert manifest["validation"]["valid"]
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["recipe"] == "pbe54/static/collinear"
    assert set(manifest["sources"]) == {"request", "poscar", "plan", "registry"}
    repeated = runner.invoke(app, args)
    assert repeated.exit_code == 2
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
