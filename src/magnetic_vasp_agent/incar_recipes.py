"""Versioned PBE input recipes with explicit moments and U/J choices."""

from __future__ import annotations

from magnetic_vasp_agent.incar import validate_incar
from magnetic_vasp_agent.incar_models import IncarBuild, IncarRecipeRequest, IncarSpec
from magnetic_vasp_agent.models import (
    CalculationTask,
    MagneticMode,
    PotentialPlan,
    PotentialRegistry,
    Structure,
)


def build_incar(
    structure: Structure,
    request: IncarRecipeRequest,
    plan: PotentialPlan,
    registry: PotentialRegistry,
) -> IncarBuild:
    request = IncarRecipeRequest.model_validate(request.model_dump())
    if request.goal != plan.goal:
        raise ValueError(
            "Recipe goal must equal the goal in the selected potential plan."
        )
    if request.goal.xc_family != "PBE_54":
        raise ValueError("Recipe version 1 supports only the reviewed PBE_54 family.")
    unknown = set(request.hubbard) - set(structure.species_order)
    if unknown:
        raise ValueError(f"Hubbard species are absent from POSCAR: {sorted(unknown)}")
    task = request.goal.task
    mode = request.goal.magnetic_mode
    spinor = mode != MagneticMode.COLLINEAR
    data = {
        "SYSTEM": structure.formula,
        "ISTART": 0,
        "ICHARG": 2,
        "ENCUT": request.encut,
        "EDIFF": request.ediff,
        "PREC": "Accurate",
        "ALGO": "Normal",
        "NELM": 300,
        "NELMIN": 8,
        "LREAL": False,
        "LASPH": True,
        "ISYM": -1,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LORBIT": 11,
        "NSW": 0,
        "IBRION": -1,
        "LWAVE": False,
        "LCHARG": True,
        "LSORBIT": mode == MagneticMode.SOC,
        "LNONCOLLINEAR": spinor,
        "MAGMOM": list(request.magmom),
        "LMAXMIX": 4,
    }
    if spinor:
        data.update(SAXIS=list(request.saxis), GGA_COMPAT=False)
    else:
        if request.saxis != [0.0, 0.0, 1.0]:
            raise ValueError(
                "Collinear recipe uses signed moments along default SAXIS; use a spinor mode for an oriented axis."
            )
        data["ISPIN"] = 2
    if task == CalculationTask.RELAX:
        data.update(NSW=200, IBRION=2, ISIF=2, EDIFFG=-0.02)
    elif task in {CalculationTask.DOS, CalculationTask.BAND}:
        data.update(ICHARG=11, LCHARG=False)
        if task == CalculationTask.DOS:
            data["NEDOS"] = 2001
        else:
            data["LORBIT"] = 12
    elif task == CalculationTask.MAGNETIC_ANISOTROPY:
        data.update(EDIFF=min(request.ediff, 1e-8), LORBMOM=True)
    if request.hubbard:
        # Lookup by element, then emit strictly in the existing species order.
        shells = [request.hubbard.get(element) for element in structure.species_order]
        data.update(
            LDAU=True,
            LDAUTYPE=2,
            LDAUL=[shell.l if shell else -1 for shell in shells],
            LDAUU=[shell.u if shell else 0.0 for shell in shells],
            LDAUJ=[shell.j if shell else 0.0 for shell in shells],
            LMAXMIX=max(4, max(2 * shell.l for shell in request.hubbard.values())),
        )
    spec = IncarSpec.model_validate(data)
    validation = validate_incar(
        spec, structure, goal=request.goal, plan=plan, registry=registry
    )
    validation.raise_for_errors()
    return IncarBuild(
        recipe=f"pbe54/{task.value}/{mode.value}",
        spec=spec,
        validation=validation,
    )
