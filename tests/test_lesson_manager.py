"""Tests for lesson management functionality."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from magnetic_vasp_agent.input_review_models import (
    InputReviewRecord,
    InputReviewStep,
    PriorLesson,
)
from magnetic_vasp_agent.lesson_manager import LessonConflict, LessonManager

if TYPE_CHECKING:
    from _pytest.tmpdir import TempPathFactory


@pytest.fixture
def temp_reviews_file(tmp_path_factory: TempPathFactory) -> Path:
    """Create a temporary reviews file with test data."""
    review_dir = tmp_path_factory.mktemp("reviews")
    reviews_file = review_dir / "input-reviews.jsonl"
    
    # Create test reviews with various scenarios
    now = datetime.now()
    test_reviews = [
        # Recent accepted review
        InputReviewRecord(
            workflow_id="wf-001",
            bundle="test-bundle",
            integrity_valid=True,
            decision="accept",
            reviewer="reviewer1",
            lesson="Use U=4.0 for FeO with LDAU",
            tags=["hubbard", "iron-oxide"],
            recorded_at=(now - timedelta(days=5)).isoformat(),
            steps=[
                InputReviewStep(
                    step_id="scf",
                    task="static",
                    magnetic_mode="collinear",
                    executable="vasp_std",
                    species_order=["Fe", "O"],
                    counts=[2, 2],
                    encut=600.0,
                    magmom=[1.0, -1.0, 0.0, 0.0],
                )
            ],
        ),
        # Recent rejected review
        InputReviewRecord(
            workflow_id="wf-002",
            bundle="test-bundle",
            integrity_valid=True,
            decision="reject",
            reviewer="reviewer2",
            lesson="KPOINTS 4x4x4 too coarse for metals",
            tags=["kpoints", "metals"],
            recorded_at=(now - timedelta(days=10)).isoformat(),
            steps=[
                InputReviewStep(
                    step_id="scf",
                    task="static",
                    magnetic_mode="none",
                    executable="vasp_std",
                    species_order=["Al"],
                    counts=[4],
                    encut=400.0,
                )
            ],
        ),
        # Old accepted review
        InputReviewRecord(
            workflow_id="wf-003",
            bundle="test-bundle",
            integrity_valid=True,
            decision="accept",
            reviewer="reviewer1",
            lesson="ISPIN=2 required for magnetic calculations",
            tags=["magnetism", "spin"],
            recorded_at=(now - timedelta(days=100)).isoformat(),
            steps=[
                InputReviewStep(
                    step_id="scf",
                    task="static",
                    magnetic_mode="collinear",
                    executable="vasp_std",
                    species_order=["Ni"],
                    counts=[2],
                    encut=500.0,
                    magmom=[2.0, 2.0],
                )
            ],
        ),
        # Conflicting review 1
        InputReviewRecord(
            workflow_id="wf-004",
            bundle="test-bundle",
            integrity_valid=True,
            decision="accept",
            reviewer="reviewer3",
            lesson="Use PREC=Accurate for oxides",
            tags=["precision", "oxides"],
            recorded_at=(now - timedelta(days=15)).isoformat(),
            steps=[
                InputReviewStep(
                    step_id="scf",
                    task="static",
                    magnetic_mode="none",
                    executable="vasp_std",
                    species_order=["Ti", "O"],
                    counts=[1, 2],
                    encut=550.0,
                )
            ],
        ),
        # Conflicting review 2 (contradicts wf-004)
        InputReviewRecord(
            workflow_id="wf-005",
            bundle="test-bundle",
            integrity_valid=True,
            decision="accept",
            reviewer="reviewer4",
            lesson="Use PREC=Normal for oxides to save time",
            tags=["precision", "oxides"],
            recorded_at=(now - timedelta(days=8)).isoformat(),
            steps=[
                InputReviewStep(
                    step_id="scf",
                    task="static",
                    magnetic_mode="none",
                    executable="vasp_std",
                    species_order=["Zn", "O"],
                    counts=[1, 1],
                    encut=400.0,
                )
            ],
        ),
    ]
    
    # Write reviews to JSONL
    with reviews_file.open("w", encoding="utf-8") as f:
        for review in test_reviews:
            f.write(review.model_dump_json() + "\n")
    
    return reviews_file


@pytest.fixture
def lesson_manager(temp_reviews_file: Path) -> LessonManager:
    """Create a LessonManager with test data."""
    return LessonManager(temp_reviews_file)


def test_load_all_lessons(lesson_manager: LessonManager) -> None:
    """Test loading all lessons from reviews file."""
    lessons = lesson_manager.load_all_lessons()
    assert len(lessons) == 5
    
    # Check that lessons have correct attributes
    assert all(hasattr(lesson, "workflow_id") for lesson in lessons)
    assert all(hasattr(lesson, "decision") for lesson in lessons)
    assert all(hasattr(lesson, "lesson") for lesson in lessons)
    assert all(hasattr(lesson, "tags") for lesson in lessons)
    assert all(hasattr(lesson, "timestamp") for lesson in lessons)


def test_load_recent_lessons(lesson_manager: LessonManager) -> None:
    """Test loading lessons within age threshold."""
    # Load lessons from last 30 days
    recent_lessons = lesson_manager.load_recent_lessons(max_age_days=30)
    assert len(recent_lessons) == 4  # Excludes wf-003 (100 days old)
    
    # All should be recent
    assert all(lesson.is_recent(max_age_days=30) for lesson in recent_lessons)
    
    # Load lessons from last 7 days
    very_recent = lesson_manager.load_recent_lessons(max_age_days=7)
    assert len(very_recent) == 1  # Only wf-001 (wf-005 is 8 days old)


def test_search_by_tags(lesson_manager: LessonManager) -> None:
    """Test searching lessons by tags."""
    # Search for single tag
    hubbard_lessons = lesson_manager.search_lessons(tags=["hubbard"])
    assert len(hubbard_lessons) == 1
    assert hubbard_lessons[0].workflow_id == "wf-001"
    
    # Search for multiple tags (OR logic)
    oxide_lessons = lesson_manager.search_lessons(tags=["oxides"])
    assert len(oxide_lessons) == 2
    assert {lesson.workflow_id for lesson in oxide_lessons} == {"wf-004", "wf-005"}


def test_search_by_decision(lesson_manager: LessonManager) -> None:
    """Test filtering lessons by decision."""
    accepted = lesson_manager.search_lessons(decision="accepted")
    assert len(accepted) == 4
    
    rejected = lesson_manager.search_lessons(decision="rejected")
    assert len(rejected) == 1
    assert rejected[0].workflow_id == "wf-002"


def test_search_by_reviewer(lesson_manager: LessonManager) -> None:
    """Test filtering lessons by reviewer."""
    reviewer1_lessons = lesson_manager.search_lessons(reviewer="reviewer1")
    assert len(reviewer1_lessons) == 2
    assert {lesson.workflow_id for lesson in reviewer1_lessons} == {"wf-001", "wf-003"}


def test_search_by_text_query(lesson_manager: LessonManager) -> None:
    """Test text search in lesson content."""
    # Search for "PREC"
    prec_lessons = lesson_manager.search_lessons(text_query="PREC")
    assert len(prec_lessons) == 2
    assert {lesson.workflow_id for lesson in prec_lessons} == {"wf-004", "wf-005"}
    
    # Search for "magnetic"
    mag_lessons = lesson_manager.search_lessons(text_query="magnetic")
    assert len(mag_lessons) == 1
    assert mag_lessons[0].workflow_id == "wf-003"


def test_search_combined_filters(lesson_manager: LessonManager) -> None:
    """Test combining multiple search filters."""
    # Recent accepted lessons with specific tag
    results = lesson_manager.search_lessons(
        tags=["oxides"],
        decision="accepted",
        max_age_days=30,
    )
    assert len(results) == 2  # Both oxide lessons are recent and accepted


def test_detect_conflicts(lesson_manager: LessonManager) -> None:
    """Test detecting conflicting lessons with same tags."""
    conflicts = lesson_manager.detect_conflicts()
    
    # Should detect conflict between wf-004 and wf-005 (both have precision+oxides tags)
    assert len(conflicts) >= 1
    
    # Find the specific conflict
    oxide_conflict = None
    for conflict in conflicts:
        if {"wf-004", "wf-005"}.issubset({lesson.workflow_id for lesson in conflict.lessons}):
            oxide_conflict = conflict
            break
    
    assert oxide_conflict is not None
    assert {"precision", "oxides"}.issubset(oxide_conflict.common_tags)
    assert len(oxide_conflict.lessons) == 2


def test_mark_deprecated(lesson_manager: LessonManager) -> None:
    """Test marking lessons as deprecated."""
    # Mark wf-004 as deprecated
    lesson_manager.mark_deprecated("wf-004", "Superseded by more recent guideline")
    
    # Reload and check
    all_lessons = lesson_manager.load_all_lessons(include_deprecated=True)
    wf_004 = next(lesson for lesson in all_lessons if lesson.workflow_id == "wf-004")
    
    assert wf_004.deprecated is True
    assert wf_004.deprecation_reason == "Superseded by more recent guideline"
    
    # Search should exclude deprecated by default
    oxide_lessons = lesson_manager.search_lessons(tags=["oxides"])
    assert len(oxide_lessons) == 1  # Only wf-005 now
    assert oxide_lessons[0].workflow_id == "wf-005"
    
    # Can include deprecated explicitly
    oxide_lessons_all = lesson_manager.search_lessons(tags=["oxides"], include_deprecated=True)
    assert len(oxide_lessons_all) == 2


def test_get_review_metrics(lesson_manager: LessonManager) -> None:
    """Test generating review metrics."""
    metrics = lesson_manager.get_review_metrics()
    
    assert metrics["total_reviews"] == 5
    assert metrics["accepted_count"] == 4
    assert metrics["rejected_count"] == 1
    assert metrics["acceptance_rate"] == 0.8
    
    # Check reviewer stats
    assert "reviewer1" in metrics["by_reviewer"]
    assert metrics["by_reviewer"]["reviewer1"]["count"] == 2
    
    # Check tag distribution
    assert "oxides" in metrics["by_tag"]
    assert metrics["by_tag"]["oxides"] == 2


def test_empty_reviews_file(tmp_path: Path) -> None:
    """Test handling empty reviews file."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()
    
    manager = LessonManager(empty_file)
    lessons = manager.load_all_lessons()
    assert len(lessons) == 0


def test_nonexistent_file(tmp_path: Path) -> None:
    """Test handling nonexistent file."""
    nonexistent = tmp_path / "does_not_exist.jsonl"
    
    manager = LessonManager(nonexistent)
    lessons = manager.load_all_lessons()
    assert len(lessons) == 0


def test_prior_lesson_is_recent(lesson_manager: LessonManager) -> None:
    """Test PriorLesson.is_recent() method."""
    lessons = lesson_manager.load_all_lessons()
    
    # Find recent and old lessons
    recent_lesson = next(lesson for lesson in lessons if lesson.workflow_id == "wf-001")
    old_lesson = next(lesson for lesson in lessons if lesson.workflow_id == "wf-003")
    
    # Test with 30 days threshold
    assert recent_lesson.is_recent(max_age_days=30) is True
    assert old_lesson.is_recent(max_age_days=30) is False
    
    # Test with 200 days threshold
    assert old_lesson.is_recent(max_age_days=200) is True
