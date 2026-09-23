"""Typed INCAR contracts; absent fields retain VASP's defaults on import."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from magnetic_vasp_agent.models import CalculationGoal

Real = Annotated[float, Field(allow_inf_nan=False, strict=True)]
Positive = Annotated[Real, Field(gt=0)]
Nonnegative = Annotated[Real, Field(ge=0)]
Integer = Annotated[int, Field(strict=True)]
PositiveInt = Annotated[Integer, Field(gt=0)]
RealList = Annotated[list[Real], Field(min_length=1, max_length=1000000)]
Axis = Annotated[list[Real], Field(min_length=3, max_length=3)]


class IncarSpec(BaseModel):
    """Supported VASP tags. Unknown tags and JSON type coercion are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    SYSTEM: str | None = None
    ISTART: Literal[0, 1, 2, 3] | None = None
    ICHARG: Literal[0, 1, 2, 4, 10, 11, 12] | None = None
    ISPIN: Literal[1, 2] | None = None
    ENCUT: Positive | None = None
    PREC: Literal["Low", "Medium", "High", "Normal", "Single", "Accurate"] | None = None
    ALGO: (
        Literal["Normal", "Fast", "VeryFast", "All", "Damped", "Exact", "None"] | None
    ) = None
    EDIFF: Nonnegative | None = None
    NELM: PositiveInt | None = None
    NELMIN: PositiveInt | None = None
    NBANDS: PositiveInt | None = None
    NCORE: PositiveInt | None = None
    NPAR: PositiveInt | None = None
    ISYM: Annotated[Integer, Field(ge=-1, le=3)] | None = None
    ISMEAR: Integer | None = None
    SIGMA: Nonnegative | None = None
    NEDOS: Annotated[Integer, Field(ge=2)] | None = None
    LORBIT: Annotated[Integer, Field(ge=0)] | None = None
    LWAVE: bool | None = None
    LCHARG: bool | None = None
    LREAL: bool | Literal["Auto", "On"] | None = None
    LSORBIT: bool | None = None
    LNONCOLLINEAR: bool | None = None
    LASPH: bool | None = None
    LORBMOM: bool | None = None
    GGA_COMPAT: bool | None = None
    LWANNIER90: bool | None = None
    ADDGRID: bool | None = None
    SAXIS: Axis | None = None
    MAGMOM: RealList | None = None
    SOCSCALE: Nonnegative | None = None
    AMIX: Nonnegative | None = None
    BMIX: Nonnegative | None = None
    AMIX_MAG: Nonnegative | None = None
    BMIX_MAG: Nonnegative | None = None
    LMAXMIX: Annotated[Integer, Field(ge=0, le=6)] | None = None
    LDAU: bool | None = None
    LDAUTYPE: Literal[1, 2, 3, 4] | None = None
    LDAUL: (
        Annotated[list[Annotated[Integer, Field(ge=-1, le=3)]], Field(min_length=1)]
        | None
    ) = None
    LDAUU: RealList | None = None
    LDAUJ: RealList | None = None
    NSW: Annotated[Integer, Field(ge=0)] | None = None
    IBRION: Annotated[Integer, Field(ge=-1, le=8)] | None = None
    ISIF: Annotated[Integer, Field(ge=0, le=7)] | None = None
    EDIFFG: Real | None = None
    POTIM: Positive | None = None

    @field_validator("SYSTEM")
    @classmethod
    def single_line_system(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.strip() or re.search(r"[\r\n;#!=\\\x00]", value)
        ):
            raise ValueError("SYSTEM must be nonempty text without INCAR delimiters")
        return value

    @field_validator("ISTART", "ICHARG", "ISPIN", "LDAUTYPE", mode="before")
    @classmethod
    def literal_integers(cls, value: object) -> object:
        if value is not None and type(value) is not int:
            raise ValueError("expected an integer, not a coerced literal")
        return value

    @field_validator("SAXIS")
    @classmethod
    def nonzero_axis(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and not any(value):
            raise ValueError("SAXIS must be nonzero")
        return value

    @property
    def spinor(self) -> bool:
        return self.LSORBIT is True or self.LNONCOLLINEAR is True

    @property
    def executable(self) -> str:
        return "vasp_ncl" if self.spinor else "vasp_std"


class IncarDiagnostic(BaseModel):
    severity: Literal["warning", "error"]
    code: str
    message: str
    tag: str | None = None
    line: int | None = None


class ParsedIncar(BaseModel):
    spec: IncarSpec
    diagnostics: list[IncarDiagnostic] = Field(default_factory=list)
    source_lines: dict[str, int] = Field(default_factory=dict)


class IncarValidation(BaseModel):
    executable: str
    required_artifacts: list[str] = Field(default_factory=list)
    diagnostics: list[IncarDiagnostic] = Field(default_factory=list)

    @computed_field
    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    def raise_for_errors(self) -> None:
        if not self.valid:
            raise ValueError(
                "; ".join(
                    f"{item.code}: {item.message}"
                    for item in self.diagnostics
                    if item.severity == "error"
                )
            )


class HubbardSpec(BaseModel):
    """Explicit U and J in eV; type 2 uses U_eff = U - J."""

    model_config = ConfigDict(extra="forbid", strict=True)
    l: Literal[0, 1, 2, 3]
    u: Nonnegative
    j: Nonnegative = 0.0

    @field_validator("l", mode="before")
    @classmethod
    def integer_l(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("l must be an integer")
        return value


class IncarRecipeSettings(BaseModel):
    """Scientific settings shared by explicit-moment and mCIF requests."""

    model_config = ConfigDict(extra="forbid")
    goal: CalculationGoal
    encut: Positive
    saxis: Axis = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    ediff: Positive = 1e-7
    hubbard: dict[str, HubbardSpec] = Field(default_factory=dict)


class IncarRecipeRequest(IncarRecipeSettings):
    """MAGMOM is already in POSCAR atom order and VASP's SAXIS spin basis."""

    magmom: RealList


class IncarBuild(BaseModel):
    recipe: str
    recipe_version: str = "1"
    spec: IncarSpec
    validation: IncarValidation
    moment_basis: str = (
        "VASP spinor basis defined by SAXIS; collinear values are signed"
    )
