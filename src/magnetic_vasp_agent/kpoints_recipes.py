"""Task-aware mesh or user-path generation without reducing magnetic cells."""

from __future__ import annotations

import math

from pymatgen.core import Lattice
from pymatgen.io.vasp.inputs import Poscar

from magnetic_vasp_agent.incar_models import IncarSpec
from magnetic_vasp_agent.kpoints import validate_kpoints
from magnetic_vasp_agent.kpoints_models import (
    KpointsBuild,
    KpointsRecipeRequest,
    KpointsSpec,
)
from magnetic_vasp_agent.models import CalculationGoal, Structure
from magnetic_vasp_agent.poscar import render_poscar


def build_kpoints(
    structure: Structure,
    request: KpointsRecipeRequest,
    incar: IncarSpec,
    *,
    incar_goal: CalculationGoal,
) -> KpointsBuild:
    request = KpointsRecipeRequest.model_validate(request.model_dump())
    if request.goal != incar_goal:
        raise ValueError(
            "KPOINTS goal must equal the INCAR request goal, including task and magnetic mode"
        )
    if request.sampling is not None:
        spec = request.sampling
        method = "user_sampling"
    else:
        if (
            not math.isfinite(structure.scale)
            or any(
                not math.isfinite(value) for row in structure.lattice for value in row
            )
            or Lattice(structure.lattice).volume <= 0
        ):
            raise ValueError("spacing generation requires a finite, nonsingular cell")
        # Poscar handles both physical scale and negative-volume scaling correctly.
        cell = Poscar.from_str(render_poscar(structure)).structure.lattice
        if not math.isfinite(cell.volume) or cell.volume <= 0:
            raise ValueError("spacing generation requires a finite, nonsingular cell")
        ratios = [length / request.spacing for length in cell.reciprocal_lattice.abc]
        if any(not math.isfinite(value) or value > 1000000 for value in ratios):
            raise ValueError("spacing produces an unsupported mesh dimension")
        mesh = [max(1, math.ceil(value)) for value in ratios]
        spec = KpointsSpec(
            mode=request.centering or "gamma",
            mesh=mesh,
            comment=f"{structure.formula}: spacing <= {request.spacing:g} 1/Angstrom (2pi basis)",
        )
        method = "reciprocal_spacing"
    validation = validate_kpoints(spec, structure, request.goal, incar)
    validation.raise_for_errors()
    return KpointsBuild(
        recipe=f"{request.goal.task.value}/{method}",
        goal=request.goal,
        spec=spec,
        validation=validation,
    )
