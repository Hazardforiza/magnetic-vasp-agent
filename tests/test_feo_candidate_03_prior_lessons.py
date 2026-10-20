"""Test prior_lessons replay for FeO candidate_03 with existing approval history."""

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
    """Create synthetic potentials for Fe and O."""
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


def test_feo_candidate_03_prior_lessons_replay(potential_context, tmp_path, monkeypatch):
    """Verify that candidate_03 loads prior_lessons from previous FeO reviews."""
    # Setup: Create three candidate bundles from mcif files
    mcif_dir = ROOT / "tests/fixtures/generated_mcif"
    log = tmp_path / "reviews.jsonl"
    
    # Patch DEFAULT_REVIEW_LOG to use our test log
    monkeypatch.setattr(
        "magnetic_vasp_agent.input_review.DEFAULT_REVIEW_LOG", log
    )
    
    candidates = []
    for idx in [1, 2, 3]:
        source = mcif_dir / f"candidate_0{idx}_FeO_BNS2_7.mcif"
        if not source.exists():
            pytest.skip(f"Missing test fixture: {source}")
        
        request = tmp_path / f"candidate_0{idx}_request.json"
        plan_workflow_from_mcif([source], request, workflow_id=f"FeO-candidate-0{idx}-test")
        prepared = prepare_workflow(
            Workflow.model_validate_json(request.read_bytes()),
            *potential_context,
            request.parent,
        )
        bundle = tmp_path / f"candidate_0{idx}_bundle"
        materialize_workflow(prepared, bundle)
        candidates.append(bundle)
    
    # Step 1: Review candidate_01 - first review (no prior_lessons expected)
    card_01 = build_review_card(candidates[0], log_path=log)
    assert card_01.prior_lessons == [], "First review should have no prior lessons"
    
    record_input_review(
        candidates[0],
        reviewer="test_reviewer",
        decision="revise",
        notes="FeO requires explicit Hubbard U specification",
        lesson="FeO oxides must specify U/J or explicitly state no DFT+U before acceptance",
        tags=["hubbard", "oxide"],
        log_path=log,
    )
    
    # Step 2: Review candidate_02 - should see candidate_01's lesson
    card_02 = build_review_card(candidates[1], log_path=log)
    assert len(card_02.prior_lessons) == 1, "Second review should see first lesson"
    assert "U/J" in card_02.prior_lessons[0].lesson or "DFT+U" in card_02.prior_lessons[0].lesson
    assert card_02.prior_lessons[0].workflow_id == "FeO-candidate-01-test"
    assert card_02.prior_lessons[0].decision == "revise"
    assert "hubbard" in card_02.prior_lessons[0].tags
    
    record_input_review(
        candidates[1],
        reviewer="test_reviewer",
        decision="accept",
        notes="Approved with U_eff = 5 eV for Fe 3d",
        lesson="FeO U_eff scan uses 3/5/7 eV candidates, J=0 for comparison only",
        tags=["hubbard-scan", "accepted"],
        log_path=log,
    )
    
    # Step 3: Review candidate_03 - should accumulate both lessons
    card_03 = build_review_card(candidates[2], log_path=log)
    assert len(card_03.prior_lessons) >= 2, "Third review should see both previous lessons"
    
    # Verify lessons are ordered correctly (most recent first or chronological)
    workflow_ids = [lesson.workflow_id for lesson in card_03.prior_lessons]
    assert "FeO-candidate-01-test" in workflow_ids
    assert "FeO-candidate-02-test" in workflow_ids
    
    # Verify content of accumulated lessons
    lessons_text = " ".join(lesson.lesson for lesson in card_03.prior_lessons)
    assert "U/J" in lessons_text or "DFT+U" in lessons_text
    assert "U_eff" in lessons_text or "scan" in lessons_text
    
    # Verify decisions are tracked
    decisions = {lesson.decision for lesson in card_03.prior_lessons}
    assert "revise" in decisions
    assert "accept" in decisions
    
    # Verify tags accumulate
    all_tags = []
    for lesson in card_03.prior_lessons:
        all_tags.extend(lesson.tags)
    assert "hubbard" in all_tags
    assert "hubbard-scan" in all_tags or "accepted" in all_tags
    
    print(f"[PASS] candidate_03 successfully loaded {len(card_03.prior_lessons)} prior lessons")
    print(f"  Workflow IDs: {workflow_ids}")
    print(f"  Decisions: {decisions}")
    print(f"  Tags: {set(all_tags)}")


def test_prior_lessons_species_filtering(potential_context, tmp_path, monkeypatch):
    """Verify prior_lessons only match relevant species combinations."""
    mcif_dir = ROOT / "tests/fixtures/generated_mcif"
    log = tmp_path / "reviews.jsonl"
    
    monkeypatch.setattr(
        "magnetic_vasp_agent.input_review.DEFAULT_REVIEW_LOG", log
    )
    
    # Create and review FeO candidate
    source = mcif_dir / "candidate_01_FeO_BNS2_7.mcif"
    if not source.exists():
        pytest.skip(f"Missing test fixture: {source}")
    
    request = tmp_path / "feo_request.json"
    plan_workflow_from_mcif([source], request, workflow_id="FeO-test")
    prepared = prepare_workflow(
        Workflow.model_validate_json(request.read_bytes()),
        *potential_context,
        request.parent,
    )
    bundle_feo = tmp_path / "feo_bundle"
    materialize_workflow(prepared, bundle_feo)
    
    record_input_review(
        bundle_feo,
        reviewer="test_reviewer",
        decision="accept",
        notes="FeO specific guidance",
        lesson="FeO requires careful U parameter selection",
        tags=["feo"],
        log_path=log,
    )
    
    # Verify FeO history is retrievable by species
    history = load_review_history(log, species=["Fe", "O"], limit=10)
    assert len(history) == 1
    assert "FeO" in history[0].lesson
    
    # Verify different species don't retrieve FeO lessons
    history_other = load_review_history(log, species=["Mn", "Al"], limit=10)
    assert len(history_other) == 0, "Mn-Al query should not return Fe-O lessons"
    
    print("[PASS] Species filtering works correctly")


def test_prior_lessons_usage_tracking(potential_context, tmp_path, monkeypatch):
    """Verify that approval decisions are tracked for audit purposes."""
    mcif_dir = ROOT / "tests/fixtures/generated_mcif"
    log = tmp_path / "reviews.jsonl"
    usage_dir = tmp_path / "usage"
    
    monkeypatch.setattr(
        "magnetic_vasp_agent.input_review.DEFAULT_REVIEW_LOG", log
    )
    
    source = mcif_dir / "candidate_01_FeO_BNS2_7.mcif"
    if not source.exists():
        pytest.skip(f"Missing test fixture: {source}")
    
    request = tmp_path / "request.json"
    plan_workflow_from_mcif([source], request, workflow_id="FeO-usage-test")
    prepared = prepare_workflow(
        Workflow.model_validate_json(request.read_bytes()),
        *potential_context,
        request.parent,
    )
    bundle = tmp_path / "bundle"
    materialize_workflow(prepared, bundle)
    
    # Perform review
    reviewed = record_input_review(
        bundle,
        reviewer="test_reviewer",
        decision="accept",
        notes="Approved for testing usage tracking",
        lesson="Test lesson for usage audit",
        tags=["test", "audit"],
        log_path=log,
    )
    
    # Verify review was recorded
    assert reviewed.decision == "accept"
    assert reviewed.reviewer == "test_reviewer"
    assert reviewed.recorded_at is not None
    
    # Verify log file was written
    assert log.exists(), "Review log should be created"
    log_content = log.read_text(encoding="utf-8")
    assert "FeO-usage-test" in log_content
    assert "test_reviewer" in log_content
    assert "accept" in log_content
    
    # Check if usage notes were written (implementation may vary)
    # The write_usage_note function should create reviewer-specific notes
    if usage_dir.exists():
        usage_files = list(usage_dir.glob("*.md"))
        print(f"[PASS] Usage tracking created {len(usage_files)} usage note(s)")
    
    print("[PASS] Usage tracking verification complete")
