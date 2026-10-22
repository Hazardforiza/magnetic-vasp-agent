from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from magnetic_vasp_agent.input_review import (
    build_review_card,
    load_review_history,
    record_input_review,
)
from magnetic_vasp_agent.mcif_workflow import plan_workflow_from_mcif
from magnetic_vasp_agent.models import PotentialPolicy, PotentialRecord, PotentialRegistry
from magnetic_vasp_agent.workflow import materialize_workflow, prepare_workflow
from magnetic_vasp_agent.workflow_models import Workflow

ROOT = Path(__file__).parents[1]


@pytest.fixture
def potential_context(tmp_path):
    records = []
    for element in ["Fe", "O"]:
        payload = f"Synthetic test potential, NOT VASP data: {element}\n".encode()
        path = tmp_path / element / "POTCAR"
        path.parent.mkdir()
        path.write_bytes(payload)
        records.append(
            PotentialRecord(
                element=element,
                potential=element,
                family="PBE_54",
                titel=f"Synthetic {element}",
                relative_path=f"{element}/POTCAR",
                sha256=hashlib.sha256(payload).hexdigest(),
                enmax=400,
            )
        )
    return PotentialRegistry(
        root=str(tmp_path), family="PBE_54", records=records
    ), PotentialPolicy(family="PBE_54", preferred={})


def test_review_card_and_human_learning_log(potential_context, tmp_path, monkeypatch):
    source = ROOT / "tests/fixtures/generated_mcif/candidate_01_FeO_BNS2_7.mcif"
    request = tmp_path / "request.json"
    plan_workflow_from_mcif([source], request, workflow_id="review-demo")
    prepared = prepare_workflow(
        Workflow.model_validate_json(request.read_bytes()),
        *potential_context,
        request.parent,
    )
    bundle = tmp_path / "bundle"
    materialize_workflow(prepared, bundle)

    default_log = tmp_path / "must-not-be-read.jsonl"
    default_log.write_text("{not-json}\n", encoding="utf-8")
    monkeypatch.setattr(
        "magnetic_vasp_agent.input_review.DEFAULT_REVIEW_LOG", default_log
    )
    log = tmp_path / "reviews.jsonl"
    card = build_review_card(bundle, log_path=log)
    assert card.awaiting_human is True
    assert card.allowed_action == "review_only"
    assert card.scientific_converged is False
    assert card.steps[0].species_order == ["Fe", "O"]
    assert card.steps[0].magmom

    with pytest.raises(ValueError, match="named human"):
        record_input_review(
            bundle,
            reviewer="agent",
            decision="accept",
            log_path=log,
        )
    with pytest.raises(ValueError, match="notes"):
        record_input_review(
            bundle,
            reviewer="cluster_user",
            decision="reject",
            log_path=log,
        )

    reviewed = record_input_review(
        bundle,
        reviewer="cluster_user",
        decision="revise",
        notes="ENCUT 600 is only a starting guess for FeO; check U before production.",
        lesson="For FeO static SCF, ask the user about DFT+U before treating ENCUT=600 as final.",
        tags=["encut", "hubbard"],
        log_path=log,
    )
    assert reviewed.awaiting_human is False
    assert reviewed.decision == "revise"
    history = load_review_history(log, species=["Fe", "O"], task="static", limit=5)
    assert len(history) == 1
    assert "DFT+U" in (history[0].lesson or "")
    assert load_review_history(log, species=["Mn", "Al"]) == []
    assert load_review_history(log, reviewer="cluster_user")
    assert load_review_history(log, reviewer="alice") == []
    usage = tmp_path / "usage" / "cluster_user.md"
    assert usage.is_file()
    usage_text = usage.read_text(encoding="utf-8")
    assert "Usage records for cluster_user" in usage_text
    assert "DFT+U" in usage_text
    assert "password=" not in usage_text

    with pytest.raises(ValueError, match="secret"):
        record_input_review(
            bundle,
            reviewer="cluster_user",
            decision="accept",
            notes="password=not-a-real-secret",
            log_path=log,
        )

    second = tmp_path / "request-2.json"
    replay = plan_workflow_from_mcif(
        [source],
        second,
        workflow_id="review-demo-2",
        review_log=log,
    )
    assert replay["prior_lessons"]
    assert "DFT+U" in replay["prior_lessons"][0]["lesson"]
    assert replay["awaiting_human"] is True
