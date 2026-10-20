"""Small, deterministic analyses over VASP calculation results.

The functions in this module consume existing ``CalculationResult`` objects
without changing the post-processing core.  Candidate records may also be
plain mappings, JSON strings, or paths to JSON produced by Pydantic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from magnetic_vasp_agent.magnetic_analysis_models import (
    CandidateEnergy,
    EnergyComparison,
    HeisenbergFit,
    HeisenbergPoint,
    MAEEntry,
    MAESummary,
    MagneticStateSummary,
    ParameterSensitivityEntry,
    ParameterSensitivitySummary,
)
from magnetic_vasp_agent.postprocess_models import CalculationResult


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def load_calculation_result(source: CalculationResult | Mapping[str, Any] | str | Path) -> CalculationResult:
    """Load a ``CalculationResult`` from an object, mapping, JSON, or path."""

    if isinstance(source, CalculationResult):
        return source
    payload: Any = source
    if isinstance(source, Path):
        payload = json.loads(source.read_text(encoding="utf-8"))
    elif isinstance(source, str):
        try:
            candidate = Path(source)
            is_file = candidate.is_file()
        except (OSError, ValueError):
            is_file = False
        payload = json.loads(candidate.read_text(encoding="utf-8") if is_file else source)
    if isinstance(payload, Mapping) and "result" in payload and "directory" not in payload:
        payload = payload["result"]
    return CalculationResult.model_validate(payload)


def _result_for(candidate: Any) -> Any:
    nested = _get(candidate, "result")
    return nested if nested is not None else candidate


def _candidate_id(candidate: Any, index: int) -> str:
    value = _get(candidate, "candidate_id", _get(candidate, "id", None))
    if value is None:
        value = _get(candidate, "name", None)
    if value is None:
        directory = _get(_result_for(candidate), "directory", None)
        value = Path(str(directory)).name if directory else f"candidate-{index + 1}"
    return str(value)


def _formula_units(candidate: Any) -> float | None:
    result = _result_for(candidate)
    for key in ("n_formula_units", "formula_units", "n_fu", "formula_unit_count"):
        value = _get(candidate, key, _get(result, key))
        if value is not None:
            try:
                value = float(value)
                if value > 0 and math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass
    return None


def _energy(candidate: Any) -> tuple[float | None, str]:
    """Return the preferred energy and its provenance.

    Sigma->0 is preferred over TOTEN/free energy and is available in either
    OUTCAR or vasprun summaries.
    """

    result = _result_for(candidate)
    direct = _get(candidate, "energy_sigma0")
    if direct is not None:
        return float(direct), "energy_sigma0"
    for owner, label in ((result, "outcar.energy_sigma0"), (_get(result, "outcar"), "outcar.energy_sigma0"), (_get(result, "vasprun"), "vasprun.final_energy_sigma0")):
        value = _get(owner, "energy_sigma0") if owner is not None else None
        if value is None and label.startswith("vasprun"):
            value = _get(owner, "final_energy_sigma0") if owner is not None else None
        if value is not None:
            return float(value), label
    for owner, key, label in (
        (candidate, "energy_per_formula_unit", "energy_per_formula_unit"),
        (candidate, "energy_per_fu", "energy_per_fu"),
        (candidate, "final_energy", "final_energy"),
        (result, "final_energy", "final_energy"),
    ):
        value = _get(owner, key)
        if value is not None:
            return float(value), label
    return None, ""


def _candidate_energy(candidate: Any, index: int, per_formula_unit: bool) -> CandidateEnergy | None:
    raw, source = _energy(candidate)
    if raw is None or not math.isfinite(raw):
        return None
    nfu = _formula_units(candidate)
    is_per_fu = source in {"energy_per_formula_unit", "energy_per_fu"}
    value = raw
    if per_formula_unit and not is_per_fu and nfu is not None:
        value = raw / nfu
        is_per_fu = True
    return CandidateEnergy(
        candidate_id=_candidate_id(candidate, index),
        energy=value,
        raw_energy=raw,
        energy_source=source,
        per_formula_unit=is_per_fu,
        n_formula_units=nfu,
    )


def compare_candidate_energies(candidates: Iterable[Any], *, per_formula_unit: bool = True) -> EnergyComparison:
    """Rank candidates by sigma->0 energy, optionally normalized per f.u."""

    entries = [item for i, candidate in enumerate(candidates) if (item := _candidate_energy(candidate, i, per_formula_unit)) is not None]
    warnings: list[str] = []
    if not entries:
        return EnergyComparison(entries=[], per_formula_unit=per_formula_unit, warnings=["no candidate has a finite energy"])
    if per_formula_unit and any(not item.per_formula_unit for item in entries):
        warnings.append("formula-unit counts are missing for some candidates; ranking is withheld to avoid mixing total-cell and per-formula-unit energies")
        return EnergyComparison(
            entries=[item.model_copy(update={"relative_energy": None}) for item in entries],
            winner_id=None,
            reference_energy=None,
            per_formula_unit=True,
            warnings=warnings,
        )
    source_classes = {
        "sigma0" if "sigma0" in item.energy_source else "fallback"
        for item in entries
    }
    if len(source_classes) > 1:
        return EnergyComparison(
            entries=[item.model_copy(update={"relative_energy": None}) for item in entries],
            winner_id=None,
            reference_energy=None,
            per_formula_unit=per_formula_unit,
            warnings=["candidates mix sigma->0 and fallback energy definitions; ranking is withheld"],
        )
    if "fallback" in source_classes:
        warnings.append("some energies fall back to final_energy because sigma->0 energy is unavailable")
    reference = min(item.energy for item in entries)
    entries.sort(key=lambda item: item.energy)
    entries = [item.model_copy(update={"relative_energy": item.energy - reference}) for item in entries]
    return EnergyComparison(entries=entries, winner_id=entries[0].candidate_id, reference_energy=reference, per_formula_unit=per_formula_unit, warnings=warnings)


def _moment_metric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return abs(float(value))
    if isinstance(value, Mapping):
        for key in ("moment", "magnitude", "total_magnetization"):
            if _get(value, key) is not None:
                return _moment_metric(_get(value, key))
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        vals = [_moment_metric(v) for v in value]
        vals = [v for v in vals if v is not None]
        return math.sqrt(sum(v * v for v in vals)) if vals else None
    return None


def _local_moment_metric(result: Any) -> float | None:
    ions = _get(result, "ion_magnetizations", []) or []
    vals = [_moment_metric(_get(ion, "moment")) for ion in ions]
    vals = [v for v in vals if v is not None]
    if vals:
        return sum(vals) / len(vals)
    return _moment_metric(_get(result, "total_magnetization")) or _moment_metric(_get(result, "total_magnetization_vector"))


def analyze_magnetic_state(result: Any, *, initial_moments: Any = None, collapse_ratio: float = 0.5, magnetic_threshold: float = 0.5) -> MagneticStateSummary:
    """Summarize a final magnetic state and flag collapse from initial moments."""

    result = _result_for(result)
    final = _local_moment_metric(result)
    if initial_moments is None:
        initial_moments = _get(result, "initial_moments")
    initial = _moment_metric(initial_moments)
    if isinstance(initial_moments, Sequence) and not isinstance(initial_moments, (str, bytes)):
        vals = [_moment_metric(v) for v in initial_moments]
        vals = [v for v in vals if v is not None]
        initial = sum(vals) / len(vals) if vals else initial
    ratio = final / initial if final is not None and initial and initial > 0 else None
    collapsed = ratio is not None and ratio <= collapse_ratio
    warnings: list[str] = []
    if collapsed:
        warnings.append(f"magnetic moment collapse detected (final/initial={ratio:.3g})")
    if final is not None and final < magnetic_threshold:
        state = "nonmagnetic_or_collapsed"
        if not collapsed:
            warnings.append(f"final magnetic moment is small ({final:.3g} uB/ion)")
    elif final is None:
        state = "unknown"
        warnings.append("final magnetic moment is unavailable")
    else:
        state = "magnetic"
    return MagneticStateSummary(state=state, initial_moment=initial, final_moment=final, collapse_ratio=ratio, collapsed=collapsed, warnings=warnings)


def summarize_mae(candidates: Iterable[Any], *, reference_direction: Any = None) -> MAESummary:
    """Summarize directional energies and identify easy/hard directions."""

    rows: list[tuple[Any, CandidateEnergy]] = []
    for index, candidate in enumerate(candidates):
        metadata = _get(candidate, "metadata", {}) or {}
        params = _get(candidate, "parameters", {}) or {}
        direction = _get(
            candidate,
            "direction",
            _get(
                candidate,
                "axis",
                _get(candidate, "saxis", _get(metadata, "direction", _get(params, "direction"))),
            ),
        )
        energy = _candidate_energy(candidate, index, False)
        if direction is not None and energy is not None:
            rows.append((direction, energy))
    warnings: list[str] = []
    if not rows:
        return MAESummary(entries=[], warnings=["no directional energies found"])
    reference = min(item.energy for _, item in rows)
    entries = [MAEEntry(direction=direction, energy=item.energy, relative_energy=item.energy - reference, candidate_id=item.candidate_id) for direction, item in rows]
    easy = min(entries, key=lambda item: item.energy)
    hard = max(entries, key=lambda item: item.energy)
    return MAESummary(entries=entries, easy_direction=easy.direction, hard_direction=hard.direction, anisotropy_energy=hard.energy - easy.energy, warnings=warnings)


def fit_heisenberg(configurations: Iterable[Any]) -> HeisenbergFit:
    """Fit ``E = intercept + J * correlation`` by ordinary least squares."""

    data: list[tuple[str, float, float]] = []
    for index, item in enumerate(configurations):
        correlation = _get(
            item,
            "correlation",
            _get(
                item,
                "spin_correlation",
                _get(
                    item,
                    "pair_correlation",
                    _get(item, "spin_product", _get(item, "s_dot_s", _get(item, "x"))),
                ),
            ),
        )
        energy, _ = _energy(item)
        if correlation is None or energy is None:
            continue
        try:
            data.append((str(_get(item, "configuration", _get(item, "name", f"config-{index + 1}"))), float(correlation), float(energy)))
        except (TypeError, ValueError):
            continue
    warnings: list[str] = []
    if len(data) < 2:
        return HeisenbergFit(warnings=["at least two configurations with correlation and energy are required"])
    mean_x = sum(row[1] for row in data) / len(data)
    mean_y = sum(row[2] for row in data) / len(data)
    denom = sum((x - mean_x) ** 2 for _, x, _ in data)
    if denom <= 0:
        return HeisenbergFit(warnings=["correlations do not span more than one value"])
    j = sum((x - mean_x) * (y - mean_y) for _, x, y in data) / denom
    intercept = mean_y - j * mean_x
    points = [HeisenbergPoint(configuration=name, correlation=x, energy=y, predicted_energy=intercept + j * x, residual=y - (intercept + j * x)) for name, x, y in data]
    ss_res = sum(point.residual**2 for point in points)
    ss_tot = sum((y - mean_y) ** 2 for _, _, y in data)
    warnings.append("single-feature slope is not a transferable J_ij without an explicit Hamiltonian mapping")
    return HeisenbergFit(coupling_j=j, intercept=intercept, points=points, residual_sum_squares=ss_res, rmse=math.sqrt(ss_res / len(points)), r_squared=(1 - ss_res / ss_tot) if ss_tot > 0 else None, warnings=warnings)


def summarize_parameter_sensitivity(candidates: Iterable[Any], parameter: str = "U") -> ParameterSensitivitySummary:
    """Group candidates by a parameter and report the lowest energy in each group."""

    groups: dict[str, tuple[Any, list[tuple[Any, CandidateEnergy]]]] = {}
    for index, candidate in enumerate(candidates):
        result = _result_for(candidate)
        params = _get(candidate, "parameters", _get(result, "parameters", {})) or {}
        value = _get(candidate, parameter, _get(params, parameter))
        item = _candidate_energy(candidate, index, True)
        if value is None or item is None:
            continue
        key = json.dumps(value, sort_keys=True, default=str)
        groups.setdefault(key, (value, []))[1].append((candidate, item))
    all_items = [item for _, rows in groups.values() for _, item in rows]
    baseline = min((item.energy for item in all_items), default=None)
    entries: list[ParameterSensitivityEntry] = []
    for value, rows in sorted(
        ((v, rows) for v, rows in groups.values()),
        key=lambda pair: (0, float(pair[0])) if isinstance(pair[0], (int, float)) else (1, repr(pair[0])),
    ):
        best = min((item for _, item in rows), key=lambda item: item.energy)
        entries.append(ParameterSensitivityEntry(parameter_value=value, best_candidate_id=best.candidate_id, best_energy=best.energy, relative_energy=best.energy - baseline if baseline is not None else None, candidate_count=len(rows)))
    best_value = min(entries, key=lambda item: item.best_energy).parameter_value if entries else None
    return ParameterSensitivitySummary(parameter=parameter, entries=entries, best_parameter_value=best_value, warnings=[] if entries else [f"no candidates contain parameter {parameter!r}"])


# Concise aliases for callers using the analysis nouns directly.
analyze_candidate_energies = compare_candidate_energies
magnetic_state_warning = analyze_magnetic_state
summarize_magnetic_state = analyze_magnetic_state
analyze_magnetic_state_warning = analyze_magnetic_state
summarize_u_sensitivity = summarize_parameter_sensitivity
fit_heisenberg_j = fit_heisenberg


__all__ = [
    "load_calculation_result", "compare_candidate_energies", "analyze_candidate_energies",
    "analyze_magnetic_state", "summarize_magnetic_state", "magnetic_state_warning", "analyze_magnetic_state_warning", "summarize_mae", "fit_heisenberg",
    "fit_heisenberg_j", "summarize_parameter_sensitivity", "summarize_u_sensitivity",
]
