"""Local input-package contracts; dependencies describe future VASP artifacts."""

from __future__ import annotations

import math
import re
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from magnetic_vasp_agent.incar_models import (
    Axis,
    IncarRecipeRequest,
    IncarSpec,
    Nonnegative,
)
from magnetic_vasp_agent.kpoints_models import KpointsRecipeRequest, StrictModel
from magnetic_vasp_agent.models import PotentialPlan


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError(
            "IDs must be 1-96 ASCII letters, digits, underscores or hyphens"
        )
    if value.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(10)),
        *(f"LPT{i}" for i in range(10)),
    }:
        raise ValueError("ID is a reserved Windows filename")
    return value


class ArtifactDependency(StrictModel):
    artifact: Literal["CHGCAR", "WAVECAR"]
    from_step: str

    _safe_source = field_validator("from_step")(safe_id)


class ExecutionCapabilities(StrictModel):
    socscale: bool = False
    evidence: str | None = None

    @model_validator(mode="after")
    def evidence_required(self) -> "ExecutionCapabilities":
        if self.socscale and (self.evidence is None or not self.evidence.strip()):
            raise ValueError("SOCSCALE support requires explicit target-build evidence")
        return self


class CalculationStep(StrictModel):
    id: str
    candidate_id: str = Field(min_length=1)
    poscar: str = Field(min_length=1)
    incar: IncarRecipeRequest
    kpoints: KpointsRecipeRequest
    potential_plan: PotentialPlan | None = None
    incar_options: IncarSpec = Field(default_factory=IncarSpec)
    dependencies: list[ArtifactDependency] = Field(default_factory=list)

    _safe_step = field_validator("id")(safe_id)

    @model_validator(mode="after")
    def consistent_step(self) -> "CalculationStep":
        if self.incar.goal != self.kpoints.goal:
            raise ValueError("step INCAR and KPOINTS goals must match")
        if self.potential_plan is not None:
            if self.potential_plan.goal != self.incar.goal:
                raise ValueError("step potential plan and request goals must match")
            if self.potential_plan.candidate_id != self.candidate_id:
                raise ValueError("step and potential plan candidate IDs must match")
        options = self.incar_options.model_dump(exclude_unset=True)
        allowed = {
            "ISTART",
            "ICHARG",
            "LWAVE",
            "LCHARG",
            "NBANDS",
            "ISMEAR",
            "SIGMA",
            "LMAXMIX",
            "SOCSCALE",
        }
        if set(options) - allowed or any(v is None for v in options.values()):
            raise ValueError(
                f"incar_options accepts non-null values only for {sorted(allowed)}"
            )
        artifacts = [d.artifact for d in self.dependencies]
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("each artifact must have exactly one source")
        return self


class ParameterScan(StrictModel):
    kind: Literal["mae", "socscale"]
    base: CalculationStep
    axes: dict[str, Axis] = Field(default_factory=dict)
    values: dict[str, Nonnegative] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_scan(self) -> "ParameterScan":
        variants = self.axes if self.kind == "mae" else self.values
        if len(variants) < 2:
            raise ValueError("a scan requires at least two variants")
        for label in variants:
            safe_id(label)
            safe_id(f"{self.base.id}-{label}")
        if self.base.dependencies:
            raise ValueError(
                "v1 scans start independently; restart dependencies are not supported"
            )
        if self.base.incar.goal.magnetic_mode.value != "soc":
            raise ValueError("MAE and SOCSCALE scans require SOC")
        if self.kind == "mae":
            if self.values or self.base.incar.goal.task.value != "magnetic_anisotropy":
                raise ValueError("MAE requires axes and magnetic_anisotropy task")
            normalized = []
            for axis in self.axes.values():
                norm = math.hypot(*axis)
                if not norm or not math.isfinite(norm):
                    raise ValueError("MAE axes must have finite nonzero length")
                normalized.append(tuple(round(x / norm, 12) for x in axis))
            if len(set(normalized)) != len(normalized):
                raise ValueError("MAE directions must be distinct after normalization")
        elif self.axes or self.base.incar.goal.task.value != "static":
            raise ValueError("SOCSCALE requires values and static task")
        elif len(set(self.values.values())) != len(self.values):
            raise ValueError("SOCSCALE values must be distinct")
        return self


class Workflow(StrictModel):
    schema_version: Literal[1] = 1
    workflow_id: str
    steps: list[CalculationStep] = Field(default_factory=list)
    scans: list[ParameterScan] = Field(default_factory=list)
    capabilities: ExecutionCapabilities = Field(default_factory=ExecutionCapabilities)

    _safe_workflow = field_validator("workflow_id")(safe_id)

    @model_validator(mode="after")
    def nonempty(self) -> "Workflow":
        if not self.steps and not self.scans:
            raise ValueError("workflow must contain at least one step or scan")
        return self


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PendingArtifact(ArtifactDependency):
    status: Literal["pending_upstream"] = "pending_upstream"
    sha256: Sha256 | None = None


class StepStatus(StrictModel):
    id: str
    input_state: Literal["prepared", "waiting_for_upstream"]
    dependencies: list[PendingArtifact]
    executable: Literal["vasp_std", "vasp_ncl"]
    runtime_verified: Literal[False] = False
