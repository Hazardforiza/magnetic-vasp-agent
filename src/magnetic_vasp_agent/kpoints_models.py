"""Typed sampling in the reciprocal basis of the actual calculation cell."""

from __future__ import annotations

import math
import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    computed_field,
    field_validator,
    model_validator,
)

from magnetic_vasp_agent.incar_models import IncarValidation
from magnetic_vasp_agent.models import CalculationGoal

Finite = Annotated[StrictFloat, Field(allow_inf_nan=False)]
Positive = Annotated[Finite, Field(gt=0)]
Vector = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Mesh = Annotated[
    list[Annotated[StrictInt, Field(ge=1, le=1000000)]],
    Field(min_length=3, max_length=3),
]
Count = Annotated[StrictInt, Field(ge=1, le=1000000)]
CoordinateMode = Literal["reciprocal", "cartesian"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PathPoint(StrictModel):
    coordinates: Vector
    label: str | None = None

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.strip() or re.search(r"[\r\n!#\x00]", value)
        ):
            raise ValueError(
                "label must be nonempty, single-line text without comment delimiters"
            )
        return value


class LineSegment(StrictModel):
    start: PathPoint
    end: PathPoint

    @model_validator(mode="after")
    def nonzero_segment(self) -> "LineSegment":
        if self.start.coordinates == self.end.coordinates:
            raise ValueError("a path segment must have distinct endpoints")
        return self


class WeightedPoint(PathPoint):
    weight: Annotated[Finite, Field(ge=0)]


class Tetrahedron(StrictModel):
    multiplicity: Count
    vertices: Annotated[list[Count], Field(min_length=4, max_length=4)]


class Tetrahedra(StrictModel):
    volume: Positive
    connections: Annotated[list[Tetrahedron], Field(min_length=1, max_length=1000000)]


class KpointsSpec(StrictModel):
    mode: Literal["gamma", "monkhorst", "line", "explicit"]
    comment: str = "Magnetic VASP Agent"
    mesh: Mesh | None = None
    shift: Vector | None = None
    coordinate_mode: CoordinateMode | None = None
    points_per_segment: Annotated[StrictInt, Field(ge=2, le=1000000)] | None = None
    segments: Annotated[list[LineSegment], Field(min_length=1)] | None = None
    points: (
        Annotated[list[WeightedPoint], Field(min_length=1, max_length=1000000)] | None
    ) = None
    tetrahedra: Tetrahedra | None = None

    @field_validator("comment")
    @classmethod
    def one_line_comment(cls, value: str) -> str:
        if re.search(r"[\r\n\x00]", value):
            raise ValueError("comment must occupy one line")
        return value

    @model_validator(mode="after")
    def consistent_mode(self) -> "KpointsSpec":
        required = {
            "gamma": {"mesh"},
            "monkhorst": {"mesh"},
            "line": {"coordinate_mode", "segments", "points_per_segment"},
            "explicit": {"coordinate_mode", "points"},
        }[self.mode]
        optional = {
            "gamma": {"shift"},
            "monkhorst": {"shift"},
            "line": set(),
            "explicit": {"tetrahedra"},
        }[self.mode]
        fields = {
            "mesh",
            "shift",
            "coordinate_mode",
            "segments",
            "points_per_segment",
            "points",
            "tetrahedra",
        }
        for field in fields:
            if field in required and getattr(self, field) is None:
                raise ValueError(f"{self.mode} mode requires {field}")
            if field not in required | optional and getattr(self, field) is not None:
                raise ValueError(f"{field} is not used in {self.mode} mode")
        if self.mode in {"gamma", "monkhorst"} and self.shift is None:
            self.shift = [0.0, 0.0, 0.0]
        if self.tetrahedra is not None:
            for tetra in self.tetrahedra.connections:
                if any(index > len(self.points) for index in tetra.vertices):
                    raise ValueError(
                        "tetrahedron indices must reference the 1-based explicit point list"
                    )
        return self

    @property
    def num_points(self) -> int:
        """Full mesh count or path sampling count, NOT irreducible-point count."""
        if self.mesh is not None:
            return math.prod(self.mesh)
        if self.segments is not None:
            return len(self.segments) * self.points_per_segment
        return len(self.points)


class KpointsDiagnostic(StrictModel):
    severity: Literal["warning", "error"]
    code: str
    message: str
    line: int | None = None


class ParsedKpoints(StrictModel):
    spec: KpointsSpec
    source_lines: dict[str, int] = Field(default_factory=dict)


class KpointsValidation(StrictModel):
    task: str
    num_points: int
    tetrahedron_check: Literal[
        "not_requested", "mesh_precheck_only", "explicit_connectivity", "failed"
    ] = "not_requested"
    incar_validation: IncarValidation
    diagnostics: list[KpointsDiagnostic] = Field(default_factory=list)

    @computed_field
    @property
    def valid(self) -> bool:
        return self.incar_validation.valid and not any(
            d.severity == "error" for d in self.diagnostics
        )

    def raise_for_errors(self) -> None:
        self.incar_validation.raise_for_errors()
        if not self.valid:
            raise ValueError(
                "; ".join(
                    f"{d.code}: {d.message}"
                    for d in self.diagnostics
                    if d.severity == "error"
                )
            )


class KpointsRecipeRequest(StrictModel):
    goal: CalculationGoal
    sampling: KpointsSpec | None = None
    spacing: Positive | None = None
    centering: Literal["gamma", "monkhorst"] | None = None

    @model_validator(mode="after")
    def sampling_choice(self) -> "KpointsRecipeRequest":
        if (self.sampling is None) == (self.spacing is None):
            raise ValueError("provide exactly one of sampling or spacing")
        if self.sampling is not None and self.centering is not None:
            raise ValueError("centering is only used with spacing")
        if self.spacing is not None and self.goal.task.value == "band":
            raise ValueError(
                "band requires an explicit Line-mode path in the calculation-cell basis"
            )
        return self


class KpointsBuild(StrictModel):
    recipe: str
    recipe_version: str = "1"
    goal: CalculationGoal
    spec: KpointsSpec
    validation: KpointsValidation
    coordinate_basis: str = (
        "Original POSCAR cell; no cell reduction, rotation, or atom reordering"
    )
