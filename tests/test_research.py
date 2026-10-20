from __future__ import annotations

import json
from pathlib import Path

from magnetic_vasp_agent.research import ResearchConfig, plan_research_from_mcif


ROOT = Path(__file__).parents[1]
MCIF = ROOT / "tests" / "fixtures" / "generated_mcif" / "candidate_01_FeO_BNS2_7.mcif"


def test_mcif_research_plan_generates_vasp_stages_and_hashes(tmp_path: Path) -> None:
    output = tmp_path / "research"
    plan = plan_research_from_mcif(
        [MCIF],
        output,
        config=ResearchConfig(
            tasks=["magnetic_order", "relax", "static", "dos", "mae", "convergence", "exchange", "phonon"],
            encut_values=[500.0, 600.0],
            kpoint_spacings=[0.30, 0.20],
        ),
    )

    assert plan.scientific_converged is False
    assert (output / "workflow.request.json").is_file()
    assert (output / "research-plan.json").is_file()
    assert plan.generated_sha256
    request = json.loads((output / "workflow.request.json").read_text(encoding="utf-8"))
    ids = [step["id"] for step in request["steps"]]
    assert any(step.startswith("relax-") for step in ids)
    assert sum(step.startswith("static-") for step in ids) >= 4
    assert any(step.startswith("dos-") for step in ids)
    assert sum(step.startswith("mae-") for step in ids) == 3
    assert sum(step.startswith("conv-") for step in ids) == 4
    stages = {stage.task: stage for stage in plan.stages}
    assert stages["exchange"].status == "requires_external_backend"
    assert stages["phonon"].status == "requires_external_backend"


def test_noncollinear_mcif_is_not_silently_rewritten_for_mae(tmp_path: Path) -> None:
    source = tmp_path / "texture.mcif"
    source.write_text(
        MCIF.read_text(encoding="utf-8").replace("2 7", "2 7"),
        encoding="utf-8",
    )
    # The fixture is collinear; the planner still reports that a noncollinear
    # MAE rotation needs an explicit physical texture when forced by the user.
    plan = plan_research_from_mcif(
        [source],
        tmp_path / "plan",
        config=ResearchConfig(tasks=["mae"], magnetic_mode="auto"),
    )
    assert any(stage.task == "mae" and stage.status == "workflow_generated" for stage in plan.stages)

