from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.incar import (
    BOOL_TAGS,
    ENUMS,
    FLOAT_TAGS,
    INT_TAGS,
    LIST_TAGS,
    IncarFormatError,
    parse_incar,
    read_incar,
    render_incar,
    validate_incar,
    write_incar,
)
from magnetic_vasp_agent.incar_models import IncarSpec
from magnetic_vasp_agent.poscar import read_poscar

FIXTURES = Path(__file__).parent / "fixtures"
HISTORY = json.loads((FIXTURES / "historical_incars.json").read_text(encoding="utf-8"))


def system_for(source: str) -> str:
    return (
        "Mn3Al"
        if source.startswith("Mn3Al")
        else ("CuMnAs" if source.startswith("CuMnAs") else "SrRuO3")
    )


def test_parser_covers_every_schema_field() -> None:
    assert set(
        IncarSpec.model_fields
    ) == BOOL_TAGS | INT_TAGS | FLOAT_TAGS | LIST_TAGS | set(ENUMS) | {
        "SYSTEM",
        "LREAL",
    }
    assert len(HISTORY) == 23


@pytest.mark.parametrize("entry", HISTORY, ids=[row["source"] for row in HISTORY])
def test_all_historical_inputs_roundtrip_and_validate(entry: dict) -> None:
    document = parse_incar(entry["text"])
    structure = read_poscar(FIXTURES / system_for(entry["source"]) / "POSCAR")
    report = validate_incar(document.spec, structure)
    assert report.valid, report.model_dump_json()
    assert (
        parse_incar(render_incar(document.spec), legacy_booleans=False).spec
        == document.spec
    )
    has_typo = ".ture." in entry["text"].lower()
    repairs = [d for d in document.diagnostics if d.code == "LEGACY_BOOLEAN"]
    assert bool(repairs) == has_typo
    if repairs:
        assert repairs[0].line == document.source_lines["LSORBIT"]
    expected_count = structure.num_atoms * (3 if document.spec.spinor else 1)
    assert len(document.spec.MAGMOM) == expected_count
    assert document.spec.ENCUT == 600


def test_preserves_afm_signs_and_srruo3_u_j() -> None:
    mn = next(e for e in HISTORY if e["source"] == "Mn3Al_nosoc/nosoc_scf/INCAR")
    assert parse_incar(mn["text"]).spec.MAGMOM == [1.4, 1.4, -2.8, 0]
    sr = next(
        e for e in HISTORY if e["source"] == "mon_scf/mon_scf_U0.6_ISMEAR-5/INCAR"
    )
    spec = parse_incar(sr["text"]).spec
    assert spec.MAGMOM == [0] * 12 + [0, 0, 6] * 4 + [0] * 36
    assert spec.LDAUL == [-1, 2, -1]
    assert spec.LDAUU == [0, 1.2, 0]
    assert spec.LDAUJ == [0, 0.6, 0]
    assert spec.SAXIS == [1, 0, 0]


def test_comments_continuations_repeats_and_exponents() -> None:
    document = parse_incar(
        "# LSORBIT=T\r\nispin=2; EDIFF=+1D-7 ! EDIFF=99\r\n"
        "MAGMOM=2*1d-3 \\ # comment after continuation\r\n"
        " -2E+0 +.0\r\nSAXIS=2*0 1; LDAUL=2*-1 2\r\n"
        "LREAL=auto; LWAVE=.F.\r\n"
    )
    assert document.spec.MAGMOM == [0.001, 0.001, -2, 0]
    assert document.spec.EDIFF == 1e-7
    assert document.spec.LDAUL == [-1, -1, 2]
    assert document.spec.LREAL == "Auto"
    assert document.spec.LSORBIT is None
    assert document.source_lines["MAGMOM"] == 3


@pytest.mark.parametrize(
    "text",
    [
        "ENCUT=600garbage",
        "ENCUT=nan",
        "ENCUT=1e999",
        "EDIFF=1e-999",
        "ISPIN=2.1",
        "ISPIN=3",
        "MAGMOM=1 -2 garbage",
        "MAGMOM=0*1",
        "MAGMOM=1000001*1",
        "MAGMOM=3*2*1",
        "MAGMOM=1 \\",
        "ENCUT=500; ENCUT=600",
        "LSORBT=T",
        "LSORBIT=.Truue.",
        "ENCUT=600; NELM",
        "ENCUT 600",
        "MAGMOM=1 2\n0 0",
        "SAXIS=0 0 0",
        "SAXIS=0 1",
        "LDAUL=1.5",
        "SYSTEM=",
        "# empty",
    ],
)
def test_malformed_input_fails_instead_of_silent_truncation(text: str) -> None:
    with pytest.raises(IncarFormatError):
        parse_incar(text)


def test_legacy_boolean_only_in_import_and_duplicate_diagnostic() -> None:
    document = parse_incar("LSORBIT=.Ture.\nLSORBIT=T")
    assert {d.code for d in document.diagnostics} == {"LEGACY_BOOLEAN", "DUPLICATE_TAG"}
    with pytest.raises(IncarFormatError, match="line 1"):
        parse_incar("LSORBIT=.Ture.", legacy_booleans=False)


@pytest.mark.parametrize(
    "data",
    [
        {"LSORBIT": "false"},
        {"ENCUT": "600"},
        {"ISPIN": True},
        {"MAGMOM": [1, float("nan")]},
        {"ENCUTT": 600},
        {"SYSTEM": "Fe\nENCUT=1"},
        {"SAXIS": [0, 0, 0]},
    ],
)
def test_json_contract_rejects_coercion_and_injection(data: dict) -> None:
    with pytest.raises(ValidationError):
        IncarSpec.model_validate(data)


def test_render_spinors_and_boolean_system_preserves_values(tmp_path: Path) -> None:
    spec = IncarSpec(
        SYSTEM="True", LSORBIT=True, MAGMOM=[0, 0, 6, 0, 0, -6], LWAVE=False
    )
    text = render_incar(spec)
    assert "LSORBIT = .TRUE." in text
    assert "SYSTEM = True" in text
    assert parse_incar(text).spec == spec
    path = tmp_path / "INCAR"
    write_incar(spec, path)
    assert read_incar(path).spec == spec
    assert b"\r\n" not in path.read_bytes()
    with pytest.raises(FileExistsError):
        write_incar(spec, path)


@pytest.mark.parametrize(
    "data,code",
    [
        ({"ISPIN": 2, "MAGMOM": [1, -1]}, "MAGMOM_LENGTH"),
        ({"LSORBIT": True, "MAGMOM": [1, 1, -2, 0]}, "MAGMOM_LENGTH"),
        ({"LNONCOLLINEAR": True}, "MAGMOM_MISSING"),
        ({"MAGMOM": [1, 1, -2, 0]}, "ISPIN_REQUIRED"),
        ({"LSORBIT": True, "LNONCOLLINEAR": False}, "SOC_CONFLICT"),
        ({"LDAU": True, "LDAUL": [2]}, "LDAU_LENGTH"),
        ({"LDAU": True}, "LDAU_MISSING"),
        ({"NELM": 5, "NELMIN": 8}, "ELECTRONIC_STEPS"),
    ],
)
def test_cross_structure_validation(data: dict, code: str) -> None:
    report = validate_incar(IncarSpec(**data), read_poscar(FIXTURES / "Mn3Al/POSCAR"))
    assert not report.valid
    assert code in {d.code for d in report.diagnostics}


def test_soc_symmetry_is_review_policy_and_binary_is_checked() -> None:
    spec = IncarSpec(LSORBIT=True, MAGMOM=[0, 0, 1] * 4, ISYM=2)
    structure = read_poscar(FIXTURES / "Mn3Al/POSCAR")
    report = validate_incar(spec, structure)
    assert report.valid
    assert "SYMMETRY_REVIEW" in {d.code for d in report.diagnostics}
    assert report.executable == "vasp_ncl"
    assert not validate_incar(spec, structure, executable="vasp_std").valid


def test_cli_import_and_invalid_validation_exit_codes(tmp_path: Path) -> None:
    path = tmp_path / "INCAR"
    path.write_text("LSORBIT=.Ture.\nMAGMOM=0 0 6", encoding="utf-8")
    runner = CliRunner()
    shown = runner.invoke(app, ["incar", "show", str(path)])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output)["spec"]["LSORBIT"] is True
    assert runner.invoke(app, ["incar", "show", str(path), "--strict"]).exit_code == 2
    checked = runner.invoke(
        app,
        ["incar", "validate", str(path), "--poscar", str(FIXTURES / "Mn3Al/POSCAR")],
    )
    assert checked.exit_code == 1
    assert json.loads(checked.output)["valid"] is False
