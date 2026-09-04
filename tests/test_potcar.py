from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from magnetic_vasp_agent.models import (
    CalculationGoal,
    CalculationTask,
    CandidateComposition,
    MagneticMode,
    PotentialPolicy,
)
from magnetic_vasp_agent.potcar import assemble_potcar, scan_library, select_by_policy


def _write_fake_potcar(root: Path, name: str, element: str, enmax: float) -> bytes:
    payload = (
        f"PAW_PBE {name}\n"
        f"   TITEL  = PAW_PBE {name} 01Jan2000\n"
        f"   VRHFIN = {element}: test\n"
        "   LEXCH  = PE\n"
        "   POMASS = 54.938; ZVAL = 7.000\n"
        f"   ENMAX  = {enmax:.3f}; ENMIN = 100.000\n"
        f"payload-{name}\n"
    ).encode("ascii")
    target = root / name / "POTCAR"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    return payload


def _policy() -> PotentialPolicy:
    return PotentialPolicy.model_validate(
        {
            "family": "PBE_54",
            "preferred": {
                "Mn": {
                    "potential": "Mn_pv",
                    "rationale": "reviewed Mn choice",
                    "confidence": 0.9,
                    "evidence": ["Mn3Al_nosoc"],
                },
                "Al": {
                    "potential": "Al",
                    "rationale": "reviewed Al choice",
                    "confidence": 0.85,
                    "evidence": ["Mn3Al_nosoc"],
                },
            },
        }
    )


def test_scan_select_and_assemble_in_species_order(tmp_path: Path) -> None:
    mn = _write_fake_potcar(tmp_path, "Mn_pv", "Mn", 270.0)
    al = _write_fake_potcar(tmp_path, "Al", "Al", 240.0)
    registry = scan_library(tmp_path, "PBE_54")

    assert len(registry.records) == 2
    mn_record = next(record for record in registry.records if record.potential == "Mn_pv")
    assert mn_record.element == "Mn"
    assert mn_record.lexch == "PE"
    assert mn_record.zval == 7.0
    assert mn_record.enmax == 270.0
    assert mn_record.sha256 == hashlib.sha256(mn).hexdigest()

    plan = select_by_policy(
        CandidateComposition(candidate_id="Mn3Al", species_order=["Mn", "Al"]),
        CalculationGoal(
            task=CalculationTask.STATIC,
            magnetic_mode=MagneticMode.COLLINEAR,
        ),
        registry,
        _policy(),
    )
    assert [choice.potential for choice in plan.choices] == ["Mn_pv", "Al"]

    output = tmp_path / "run" / "POTCAR"
    manifest = assemble_potcar(plan, registry, output)
    assert output.read_bytes() == mn + al
    assert manifest.potential_names == ["Mn_pv", "Al"]
    assert manifest.output_sha256 == hashlib.sha256(mn + al).hexdigest()


def test_assemble_rejects_source_changed_after_inventory(tmp_path: Path) -> None:
    _write_fake_potcar(tmp_path, "Mn_pv", "Mn", 270.0)
    _write_fake_potcar(tmp_path, "Al", "Al", 240.0)
    registry = scan_library(tmp_path, "PBE_54")
    plan = select_by_policy(
        CandidateComposition(candidate_id="Mn3Al", species_order=["Mn", "Al"]),
        CalculationGoal(
            task=CalculationTask.STATIC,
            magnetic_mode=MagneticMode.COLLINEAR,
        ),
        registry,
        _policy(),
    )

    (tmp_path / "Mn_pv" / "POTCAR").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after inventory"):
        assemble_potcar(plan, registry, tmp_path / "run" / "POTCAR")


def test_selector_rejects_missing_reviewed_potential(tmp_path: Path) -> None:
    _write_fake_potcar(tmp_path, "Mn", "Mn", 270.0)
    _write_fake_potcar(tmp_path, "Al", "Al", 240.0)
    registry = scan_library(tmp_path, "PBE_54")

    with pytest.raises(ValueError, match="Mn_pv.*absent"):
        select_by_policy(
            CandidateComposition(candidate_id="Mn3Al", species_order=["Mn", "Al"]),
            CalculationGoal(
                task=CalculationTask.STATIC,
                magnetic_mode=MagneticMode.COLLINEAR,
            ),
            registry,
            _policy(),
        )
