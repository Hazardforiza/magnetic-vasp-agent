"""Models returned by :mod:`magnetic_vasp_agent.magnetic_analysis`.

The models intentionally contain compact, JSON-friendly summaries.  They do
not replace the auditable post-processing models in ``postprocess_models``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CandidateEnergy(BaseModel):
    candidate_id: str
    energy: float
    raw_energy: float
    energy_source: str
    per_formula_unit: bool = False
    n_formula_units: float | None = None
    relative_energy: float | None = 0.0


class EnergyComparison(BaseModel):
    entries: list[CandidateEnergy]
    winner_id: str | None = None
    reference_energy: float | None = None
    per_formula_unit: bool = True
    warnings: list[str] = Field(default_factory=list)


class MagneticStateSummary(BaseModel):
    state: str
    initial_moment: float | None = None
    final_moment: float | None = None
    collapse_ratio: float | None = None
    collapsed: bool = False
    warnings: list[str] = Field(default_factory=list)


class MAEEntry(BaseModel):
    direction: Any
    energy: float
    relative_energy: float
    candidate_id: str


class MAESummary(BaseModel):
    entries: list[MAEEntry]
    easy_direction: Any = None
    hard_direction: Any = None
    anisotropy_energy: float | None = None
    warnings: list[str] = Field(default_factory=list)


class HeisenbergPoint(BaseModel):
    configuration: str
    correlation: float
    energy: float
    predicted_energy: float
    residual: float


class HeisenbergFit(BaseModel):
    model: str = "E = intercept + J * correlation"
    coupling_j: float | None = None
    intercept: float | None = None
    points: list[HeisenbergPoint] = Field(default_factory=list)
    residual_sum_squares: float | None = None
    rmse: float | None = None
    r_squared: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ParameterSensitivityEntry(BaseModel):
    parameter_value: Any
    best_candidate_id: str | None = None
    best_energy: float | None = None
    relative_energy: float | None = None
    candidate_count: int = 0


class ParameterSensitivitySummary(BaseModel):
    parameter: str
    entries: list[ParameterSensitivityEntry]
    best_parameter_value: Any = None
    warnings: list[str] = Field(default_factory=list)
