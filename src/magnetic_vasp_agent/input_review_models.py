"""Human review records for generated VASP input packages."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from magnetic_vasp_agent.knowledge.models import StrictModel

REVIEWER_BLOCKLIST = {"agent", "llm", "model", "assistant", "chatgpt", "auto"}


class PriorLesson(StrictModel):
    workflow_id: str
    decision: str
    reviewer: str | None = None
    lesson: str
    tags: list[str] = Field(default_factory=list)


class InputReviewStep(StrictModel):
    step_id: str
    task: str
    magnetic_mode: str
    executable: str
    species_order: list[str]
    counts: list[int]
    encut: float | None = None
    magmom: list[float] = Field(default_factory=list)
    potcar_names: list[str] = Field(default_factory=list)
    kpoints_summary: str | None = None
    file_hashes: dict[str, str] = Field(default_factory=dict)


class InputReviewCard(StrictModel):
    schema_version: Literal[1] = 1
    workflow_id: str
    bundle: str
    integrity_valid: bool
    scientific_converged: Literal[False] = False
    awaiting_human: bool = True
    allowed_action: Literal["review_only"] = "review_only"
    steps: list[InputReviewStep]
    status: Literal["pending", "reviewed"] = "pending"
    decision: Literal["accept", "reject", "revise"] | None = None
    reviewer: str | None = None
    notes: str | None = None
    lesson: str | None = None
    tags: list[str] = Field(default_factory=list)
    prior_lessons: list[PriorLesson] = Field(default_factory=list)
    recorded_at: str | None = None

    @field_validator("reviewer")
    @classmethod
    def human_reviewer(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip() or value.strip().casefold() in REVIEWER_BLOCKLIST:
            raise ValueError("reviewer must be a named human, not the agent")
        return value.strip()


class InputReviewRecord(InputReviewCard):
    """Append-only learning record copied from a finished review."""
