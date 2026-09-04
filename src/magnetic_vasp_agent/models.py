from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class CalculationTask(StrEnum):
    RELAX = "relax"
    STATIC = "static"
    DOS = "dos"
    BAND = "band"
    MAGNETIC_ANISOTROPY = "magnetic_anisotropy"


class MagneticMode(StrEnum):
    COLLINEAR = "collinear"
    NONCOLLINEAR = "noncollinear"
    SOC = "soc"


class CalculationGoal(BaseModel):
    task: CalculationTask
    magnetic_mode: MagneticMode
    xc_family: str = "PBE_54"
    notes: str | None = None

    @model_validator(mode="after")
    def validate_task_mode(self) -> "CalculationGoal":
        if (
            self.task == CalculationTask.MAGNETIC_ANISOTROPY
            and self.magnetic_mode != MagneticMode.SOC
        ):
            raise ValueError("magnetic_anisotropy requires magnetic_mode='soc'")
        return self


class CandidateComposition(BaseModel):
    candidate_id: str = Field(min_length=1)
    species_order: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_species_order(self) -> "CandidateComposition":
        if len(set(self.species_order)) != len(self.species_order):
            raise ValueError("species_order must contain unique POSCAR species")
        return self


class PotentialRecord(BaseModel):
    element: str = Field(pattern=r"^[A-Z][a-z]?$")
    potential: str = Field(min_length=1)
    family: str = Field(min_length=1)
    titel: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vrhfin: str | None = None
    lexch: str | None = None
    zval: float | None = None
    enmax: float | None = None


class PotentialRegistry(BaseModel):
    schema_version: int = 1
    family: str
    root: str
    records: list[PotentialRecord]


class PreferredPotential(BaseModel):
    potential: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class PotentialPolicy(BaseModel):
    schema_version: int = 1
    family: str
    preferred: dict[str, PreferredPotential]


class PotentialChoice(BaseModel):
    element: str = Field(pattern=r"^[A-Z][a-z]?$")
    potential: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class PotentialPlan(BaseModel):
    schema_version: int = 1
    candidate_id: str
    species_order: list[str] = Field(min_length=1)
    goal: CalculationGoal
    choices: list[PotentialChoice] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_choice_order(self) -> "PotentialPlan":
        choice_elements = [choice.element for choice in self.choices]
        if choice_elements != self.species_order:
            raise ValueError(
                "potential choices must follow the exact POSCAR species order"
            )
        return self


class PotcarBuildManifest(BaseModel):
    schema_version: int = 1
    candidate_id: str
    family: str
    species_order: list[str]
    potential_names: list[str]
    source_sha256: list[str]
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
