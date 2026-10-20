from __future__ import annotations

import json

import pytest

from magnetic_vasp_agent.magnetic_analysis import (
    analyze_magnetic_state,
    compare_candidate_energies,
    fit_heisenberg,
    load_calculation_result,
    summarize_mae,
    summarize_parameter_sensitivity,
)
from magnetic_vasp_agent.postprocess_models import (
    CalculationResult,
    DetectedMagneticMode,
    IonMagnetization,
    OutcarParameters,
    OutcarSummary,
)


def _result(energy: float, sigma0: float, moments: list[float]) -> CalculationResult:
    outcar = OutcarSummary(
        parameters=OutcarParameters(ispin=2, lnoncollinear=False, lsorbit=False, nelm=60, nsw=0, nions=len(moments)),
        magnetic_mode=DetectedMagneticMode.COLLINEAR,
        electronic_converged=True,
        free_energy_toten=energy,
        energy_sigma0=sigma0,
        ion_magnetizations=[IonMagnetization(index=i + 1, moment=m) for i, m in enumerate(moments)],
        source_sha256="0" * 64,
    )
    return CalculationResult(directory="/tmp/calc", magnetic_mode=DetectedMagneticMode.COLLINEAR, converged=True, final_energy=energy, outcar=outcar, ion_magnetizations=outcar.ion_magnetizations)


def test_load_calculation_result_accepts_object_and_json() -> None:
    result = _result(-10.0, -10.2, [3.0, -3.0])
    assert load_calculation_result(result) is result
    loaded = load_calculation_result(result.model_dump_json())
    assert loaded.outcar is not None
    assert loaded.outcar.energy_sigma0 == pytest.approx(-10.2)


def test_compare_candidates_prefers_sigma0_and_normalizes_per_formula_unit() -> None:
    result = _result(-20.0, -20.4, [3.0, -3.0])
    comparison = compare_candidate_energies([
        {"candidate_id": "a", "result": result, "n_formula_units": 2},
        {"candidate_id": "b", "energy_sigma0": -9.9, "n_formula_units": 1},
    ])
    assert comparison.winner_id == "a"
    assert comparison.entries[0].energy == pytest.approx(-10.2)
    assert comparison.entries[0].energy_source == "outcar.energy_sigma0"


def test_compare_withholds_ranking_when_formula_unit_normalization_is_incomplete() -> None:
    comparison = compare_candidate_energies([
        {"candidate_id": "cell2", "energy_sigma0": -20.0, "n_formula_units": 2},
        {"candidate_id": "unknown", "energy_sigma0": -11.0},
    ])
    assert comparison.winner_id is None
    assert comparison.reference_energy is None
    assert all(item.relative_energy is None for item in comparison.entries)
    assert comparison.warnings


def test_magnetic_state_flags_collapse() -> None:
    summary = analyze_magnetic_state(_result(-1.0, -1.0, [0.2, -0.1]), initial_moments=[3.0, -3.0])
    assert summary.collapsed is True
    assert summary.state == "nonmagnetic_or_collapsed"
    assert summary.warnings


def test_mae_and_heisenberg_fit_report_residuals() -> None:
    mae = summarize_mae([
        {"candidate_id": "z", "direction": [0, 0, 1], "energy_sigma0": -10.0},
        {"candidate_id": "x", "direction": [1, 0, 0], "energy_sigma0": -9.8},
    ])
    assert mae.easy_direction == [0, 0, 1]
    assert mae.anisotropy_energy == pytest.approx(0.2)

    fit = fit_heisenberg([
        {"configuration": "FM", "correlation": 1, "energy_sigma0": -2.0},
        {"configuration": "AFM", "correlation": -1, "energy_sigma0": 0.0},
        {"configuration": "mix", "correlation": 0, "energy_sigma0": -1.0},
    ])
    assert fit.coupling_j == pytest.approx(-1.0)
    assert fit.rmse == pytest.approx(0.0)
    assert all(point.residual == pytest.approx(0.0) for point in fit.points)


def test_parameter_sensitivity_groups_u_values() -> None:
    summary = summarize_parameter_sensitivity([
        {"candidate_id": "u2", "parameters": {"U": 2}, "energy_sigma0": -5.0},
        {"candidate_id": "u4", "parameters": {"U": 4}, "energy_sigma0": -5.3},
    ])
    assert summary.best_parameter_value == 4
    assert [entry.relative_energy for entry in summary.entries] == pytest.approx([0.3, 0.0])
