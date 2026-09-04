from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

ELEMENT_PATTERN = r"^[A-Z][a-z]?$"

Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Flags3 = Annotated[list[bool], Field(min_length=3, max_length=3)]


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


class CoordinateMode(StrEnum):
    DIRECT = "direct"
    CARTESIAN = "cartesian"


class Structure(BaseModel):
    """A crystal structure in POSCAR terms: species blocks in POSCAR order."""

    comment: str = ""
    scale: float
    lattice: list[Vector3] = Field(min_length=3, max_length=3)
    species_order: list[str] = Field(min_length=1)
    counts: list[int] = Field(min_length=1)
    coordinate_mode: CoordinateMode
    positions: list[Vector3]
    selective_dynamics: list[Flags3] | None = None
    velocities: list[Vector3] | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> "Structure":
        if self.scale == 0:
            raise ValueError("POSCAR scale factor must be non-zero")
        _validate_species(self.species_order)
        if len(self.counts) != len(self.species_order):
            raise ValueError(
                f"{len(self.counts)} atom counts for {len(self.species_order)} species"
            )
        if any(count <= 0 for count in self.counts):
            raise ValueError("every species count must be positive")
        expected = sum(self.counts)
        if len(self.positions) != expected:
            raise ValueError(
                f"{len(self.positions)} positions for {expected} atoms"
            )
        if self.selective_dynamics is not None and len(self.selective_dynamics) != expected:
            raise ValueError("selective dynamics flags must cover every atom")
        if self.velocities is not None and len(self.velocities) != expected:
            raise ValueError("velocities must cover every atom")
        return self

    @property
    def num_atoms(self) -> int:
        return sum(self.counts)

    @property
    def site_species(self) -> list[str]:
        """Species symbol for every atom, expanded in POSCAR order."""
        return [
            species
            for species, count in zip(self.species_order, self.counts, strict=True)
            for _ in range(count)
        ]

    @property
    def formula(self) -> str:
        return "".join(
            f"{species}{count if count != 1 else ''}"
            for species, count in zip(self.species_order, self.counts, strict=True)
        )


def _validate_species(species_order: list[str]) -> None:
    for species in species_order:
        if not re.fullmatch(ELEMENT_PATTERN, species):
            raise ValueError(f"species {species!r} is not an element symbol")
    if len(set(species_order)) != len(species_order):
        raise ValueError("species_order must contain unique POSCAR species")


class CandidateComposition(BaseModel):
    candidate_id: str = Field(min_length=1)
    species_order: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_species_order(self) -> "CandidateComposition":
        _validate_species(self.species_order)
        return self

    @classmethod
    def from_structure(
        cls, structure: Structure, candidate_id: str | None = None
    ) -> "CandidateComposition":
        return cls(
            candidate_id=candidate_id or structure.formula,
            species_order=list(structure.species_order),
        )


class PotentialRecord(BaseModel):
    element: str = Field(pattern=ELEMENT_PATTERN)
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
    element: str = Field(pattern=ELEMENT_PATTERN)
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
