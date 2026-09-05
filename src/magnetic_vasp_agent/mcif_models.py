"""Traceable magnetic candidates in the unmodified magnetic calculation cell."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from magnetic_vasp_agent.incar_models import IncarRecipeSettings, RealList
from magnetic_vasp_agent.kpoints_models import StrictModel, Vector
from magnetic_vasp_agent.models import Structure


class SiteOrigin(StrictModel):
    source_label: str
    source_row: int
    operation_id: str
    centering_id: str


class MagneticSite(StrictModel):
    poscar_index: int
    species: str
    fractional: Vector
    moment_cartesian: Vector
    origins: list[SiteOrigin]


class SourceSite(StrictModel):
    label: str
    row: int
    species: str
    fractional: Vector
    moment_crystalaxis: Vector
    moment_cartesian: Vector
    occupancy: float
    occupancy_explicit: bool
    wyckoff_declared: str | None


class MagneticOperation(StrictModel):
    operation_id: str
    centering_id: str
    operation: str
    centering: str
    combined: str


class CandidateStructure(StrictModel):
    schema_version: Literal[1] = 1
    candidate_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_block: str
    declared_bns: str | None = None
    declared_space_group: str | None = None
    symmetry_status: Literal["explicit_operations_validated_not_bns_identified"] = (
        "explicit_operations_validated_not_bns_identified"
    )
    structure: Structure
    source_sites: list[SourceSite]
    magnetic_operations: list[MagneticOperation]
    sites: list[MagneticSite]
    collinear: bool
    collinear_axis_cartesian: Vector | None
    total_moment_cartesian: Vector
    diagnostics: list[str]
    coordinate_basis: str = "pymatgen Lattice.from_parameters Cartesian frame; original magnetic cell, no reduction"

    @field_validator("candidate_id")
    @classmethod
    def single_line_id(cls, value: str) -> str:
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("candidate ID must be nonempty single-line text")
        return value

    @model_validator(mode="after")
    def aligned_sites(self) -> "CandidateStructure":
        if len(self.sites) != self.structure.num_atoms:
            raise ValueError("candidate site mapping does not cover the structure")
        if (
            self.structure.scale != 1
            or self.structure.coordinate_mode.value != "direct"
        ):
            raise ValueError(
                "candidate structure requires unit scale and fractional coordinates"
            )
        if any(not math.isfinite(x) for row in self.structure.lattice for x in row):
            raise ValueError("candidate lattice must be finite")
        for i, site in enumerate(self.sites):
            if (
                site.poscar_index != i + 1
                or site.species != self.structure.site_species[i]
                or site.fractional != self.structure.positions[i]
            ):
                raise ValueError(
                    "candidate site mapping and POSCAR atom order disagree"
                )
        return self


class McifIncarRequest(IncarRecipeSettings):
    moment_source: Literal["mcif"]
    moment_mode: Literal["preserve_cartesian", "collinear_projection"] = (
        "preserve_cartesian"
    )


class MomentMapping(StrictModel):
    mode: Literal["preserve_cartesian", "collinear_projection"]
    magmom: RealList
    saxis: Vector
    projection_axis_cartesian: Vector | None = None
    orientation_preserved: bool
    maximum_roundtrip_error: float
