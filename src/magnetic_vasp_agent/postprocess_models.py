"""Typed models for VASP post-processing results.

These mirror the deterministic-core style: plain :class:`pydantic.BaseModel`
records with explicit validators. Every parser records the SHA256 of the source
file it read, and the combined :class:`CalculationResult` carries a
``result_sha256`` over its own canonical JSON so a result is auditable and
refuses silent drift. No raw arrays (full DOS, band eigenvalues) are stored;
only compact, decision-relevant summaries.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DetectedMagneticMode(StrEnum):
    """Magnetic treatment inferred from the run's own flags/output."""

    NONMAGNETIC = "nonmagnetic"  # ISPIN=1
    COLLINEAR = "collinear"  # ISPIN=2, not noncollinear
    NONCOLLINEAR = "noncollinear"  # LNONCOLLINEAR=T, LSORBIT=F
    SOC = "soc"  # LSORBIT=T
    UNKNOWN = "unknown"


class IonMagnetization(BaseModel):
    """Local magnetic moment on a single ion (1-based index)."""

    index: int = Field(ge=1)
    species: str | None = None
    # Collinear: signed scalar moment (the OUTCAR "tot" column).
    moment: float | None = None
    # Noncollinear/SOC: [mx, my, mz]; ``moment`` then holds the magnitude.
    vector: list[float] | None = Field(default=None, min_length=3, max_length=3)
    # Orbital-resolved totals from the OUTCAR block, e.g. {"s":.., "p":.., "d":..}.
    orbital_moments: dict[str, float] | None = None

    @model_validator(mode="after")
    def validate_moment(self) -> IonMagnetization:
        if self.moment is None and self.vector is None:
            raise ValueError("an ion magnetization needs a scalar or a vector")
        return self


class OszicarSummary(BaseModel):
    """What the OSZICAR tells us: SCF cadence, final energy, and net moment."""

    ionic_steps: int = Field(ge=0)
    electronic_steps_total: int = Field(ge=0)
    electronic_steps_last: int = Field(ge=0)
    final_free_energy: float | None = None  # the "F=" column
    final_e0: float | None = None  # the "E0=" column
    final_delta_e: float | None = None  # the "d E =" column
    total_magnetization: float | None = None  # collinear "mag="
    magnetization_vector: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )  # noncollinear/SOC "mag=" (three components)
    source_sha256: str = Field(pattern=SHA256_PATTERN)


class OutcarParameters(BaseModel):
    """The handful of OUTCAR control flags that define the magnetic run."""

    ispin: int = Field(ge=1, le=2)
    lnoncollinear: bool
    lsorbit: bool
    nelm: int = Field(ge=1)
    nsw: int = Field(ge=0)
    nions: int = Field(ge=1)
    nelect: float | None = None


class OutcarSummary(BaseModel):
    """Scientifically critical quantities parsed from the OUTCAR."""

    parameters: OutcarParameters
    magnetic_mode: DetectedMagneticMode
    electronic_converged: bool  # "reached required accuracy"
    free_energy_toten: float | None = None
    energy_without_entropy: float | None = None
    energy_sigma0: float | None = None
    fermi_energy: float | None = None
    ion_magnetizations: list[IonMagnetization] = Field(default_factory=list)
    total_magnetization: float | None = None
    total_magnetization_vector: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )
    max_force: float | None = None  # max |F| on any atom (eV/A)
    source_sha256: str = Field(pattern=SHA256_PATTERN)


class DosSummary(BaseModel):
    """A compact density-of-states summary (never the full arrays)."""

    efermi: float
    is_spin_polarized: bool
    dos_at_fermi: list[float] = Field(min_length=1, max_length=2)  # per spin channel
    estimated_gap: float | None = None  # crude tDOS gap around E_F, or None if metallic


class VasprunSummary(BaseModel):
    """Structured cross-check from vasprun.xml (memory-safe streaming parse)."""

    vasp_version: str | None = None
    ispin: int = Field(ge=1, le=2)
    noncollinear: bool = False
    soc: bool = False
    electronic_converged: bool | None = None
    ionic_converged: bool | None = None
    final_free_energy: float | None = None
    final_energy_sigma0: float | None = None
    fermi_energy: float | None = None
    total_magnetization: float | None = None
    dos: DosSummary | None = None
    source_sha256: str = Field(pattern=SHA256_PATTERN)


class CrossCheck(BaseModel):
    """Agreement of one quantity across the files that reported it."""

    quantity: str
    sources: dict[str, float]
    max_abs_diff: float = Field(ge=0.0)
    tolerance: float = Field(ge=0.0)
    consistent: bool


class CalculationResult(BaseModel):
    """The combined, auditable result for one calculation directory."""

    schema_version: int = 1
    directory: str
    magnetic_mode: DetectedMagneticMode
    converged: bool
    final_energy: float | None = None  # best available (OUTCAR TOTEN preferred)
    energy_per_atom: float | None = None
    fermi_energy: float | None = None
    total_magnetization: float | None = None
    total_magnetization_vector: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )
    ion_magnetizations: list[IonMagnetization] = Field(default_factory=list)
    max_force: float | None = None
    oszicar: OszicarSummary | None = None
    outcar: OutcarSummary | None = None
    vasprun: VasprunSummary | None = None
    cross_checks: list[CrossCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    result_sha256: str = Field(default="", pattern=r"^([0-9a-f]{64})?$")

    @model_validator(mode="after")
    def validate_sources(self) -> CalculationResult:
        if self.oszicar is None and self.outcar is None and self.vasprun is None:
            raise ValueError(
                "a calculation result needs at least one parsed source file"
            )
        return self
