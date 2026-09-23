import pytest
from pydantic import ValidationError

from magnetic_vasp_agent.models import (
    CalculationGoal,
    CalculationTask,
    MagneticMode,
    PotentialChoice,
    PotentialPlan,
)


def test_magnetic_anisotropy_requires_soc() -> None:
    with pytest.raises(ValidationError, match="requires magnetic_mode='soc'"):
        CalculationGoal(
            task=CalculationTask.MAGNETIC_ANISOTROPY,
            magnetic_mode=MagneticMode.COLLINEAR,
        )

def test_plan_rejects_choice_order_mismatch() -> None:
    with pytest.raises(ValidationError, match="exact POSCAR species order"):
        PotentialPlan(
            candidate_id="Mn3Al",
            species_order=["Mn", "Al"],
            goal=CalculationGoal(
                task=CalculationTask.STATIC,
                magnetic_mode=MagneticMode.COLLINEAR,
            ),
            choices=[
                PotentialChoice(
                    element="Al",
                    potential="Al",
                    rationale="test",
                    confidence=1,
                ),
                PotentialChoice(
                    element="Mn",
                    potential="Mn_pv",
                    rationale="test",
                    confidence=1,
                ),
            ],
        )
