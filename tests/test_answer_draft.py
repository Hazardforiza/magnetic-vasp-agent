import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from magnetic_vasp_agent.cli import app
from magnetic_vasp_agent.knowledge.retrieval import (
    annotate_answer_draft_conflict,
    build_answer_draft,
    evidence_package_sha256,
    record_answer_draft_review,
    validate_answer_draft,
)
from magnetic_vasp_agent.knowledge.retrieval_models import (
    CitedEvidence,
    EvidencePackage,
    RetrievalFilter,
)
from magnetic_vasp_agent.knowledge.models import KnowledgeSourceType


def evidence_package(*, status: str = "evidence_ready") -> EvidencePackage:
    citation = CitedEvidence(
        rank=1,
        score=1.0,
        document_id="doc-1",
        source_id="vasp-wiki-magmom",
        source_type=KnowledgeSourceType.OFFICIAL,
        authority=100,
        source_validated=True,
        title="MAGMOM",
        section_path=["MAGMOM"],
        quote="MAGMOM sets initial magnetic moments.",
        text_sha256="a" * 64,
        citation_id="E1",
    )
    return EvidencePackage(
        schema_version=1,
        question="How do I set MAGMOM?",
        retrieval_backend="bm25",
        filters=RetrievalFilter(),
        index_manifest_sha256="b" * 64,
        citations=[citation] if status != "evidence_missing" else [],
        required_source_types=[],
        source_types_present=[KnowledgeSourceType.OFFICIAL] if status != "evidence_missing" else [],
        missing_source_types=[],
        risks=[],
        status=status,
        human_review_required=status != "evidence_ready",
        answering_rules=["Every technical claim must cite evidence."],
    )


def test_build_and_validate_cited_draft():
    package = evidence_package()
    draft = build_answer_draft(package, ["Set initial moments with MAGMOM. [E1]"])
    assert draft.status == "draft_ready_for_review"
    assert draft.allowed_action == "draft_only"
    assert draft.human_review_required is True
    assert draft.claims[0].citation_ids == ["E1"]
    assert draft.citation_integrity.citation_complete
    assert validate_answer_draft(draft, package).citation_complete
    assert len(evidence_package_sha256(package)) == 64


def test_unknown_and_uncited_claims_require_review():
    package = evidence_package()
    draft = build_answer_draft(package, ["Supported statement. [E9]", "No source here"])
    assert draft.status == "review_required"
    assert draft.citation_integrity.unknown_citation_ids == ["E9"]
    assert draft.citation_integrity.uncited_claim_ids == ["C2"]
    assert any(reason.code == "citation_incomplete" for reason in draft.review_reasons)
    validate_answer_draft(draft, package)


def test_package_review_status_propagates_to_draft():
    package = evidence_package(status="review_required")
    draft = build_answer_draft(package, ["Set initial moments. [E1]"])
    assert draft.status == "review_required"
    assert any(reason.code == "evidence_package_review_required" for reason in draft.review_reasons)


def test_draft_rejects_tampered_integrity_or_package_hash():
    package = evidence_package()
    draft = build_answer_draft(package, ["Set initial moments. [E1]"])
    tampered = draft.model_copy(update={"evidence_package_sha256": "c" * 64})
    with pytest.raises(ValueError, match="evidence package hash mismatch"):
        validate_answer_draft(tampered, package)


def test_draft_binding_rejects_question_or_status_drift():
    package = evidence_package()
    draft = build_answer_draft(package, ["Set initial moments. [E1]"])
    with pytest.raises(ValueError, match="question does not match"):
        validate_answer_draft(draft.model_copy(update={"question": " 다른问题"}), package)
    with pytest.raises(ValueError, match="status does not match"):
        validate_answer_draft(
            draft.model_copy(update={"evidence_package_status": "review_required"}),
            package,
        )


def test_claim_markers_must_match_citation_ids():
    with pytest.raises(ValidationError, match="citation_ids"):
        from magnetic_vasp_agent.knowledge.retrieval_models import AnswerDraftClaim

        AnswerDraftClaim(claim_id="C1", text="Claim [E1]", citation_ids=[])


def test_knowledge_draft_cli_writes_review_only_artifact(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        evidence_package(status="review_required").model_dump_json(), encoding="utf-8"
    )
    output = tmp_path / "draft.json"
    result = CliRunner().invoke(
        app,
        [
            "knowledge",
            "draft",
            "--evidence",
            str(evidence),
            "--output",
            str(output),
            "--claim",
            "Set initial moments with MAGMOM. [E1]",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = output.read_text(encoding="utf-8")
    assert '"status": "review_required"' in payload
    assert '"allowed_action": "draft_only"' in payload

    repeated = CliRunner().invoke(
        app,
        ["knowledge", "draft", "--evidence", str(evidence), "--output", str(output)],
    )
    assert repeated.exit_code == 2
    assert "answer draft exists" in repeated.output


def test_conflict_annotation_and_review_remain_non_executable():
    package = evidence_package()
    package = package.model_copy(update={
        "citations": [
            package.citations[0],
            package.citations[0].model_copy(update={"citation_id": "E2", "quote": "A different rule."}),
        ]
    })
    draft = build_answer_draft(package, ["Use the documented rule. [E1] [E2]"])
    annotated = annotate_answer_draft_conflict(
        draft, package, citation_ids=["E1", "E2"], description="Sources disagree on applicability."
    )
    assert annotated.status == "review_required"
    assert annotated.conflicts[0].conflict_id == "CF1"
    assert annotated.review.status == "review_required"
    reviewed = record_answer_draft_review(
        annotated, package, reviewer="alice", decision="needs_revision",
        notes="Check target VASP version.",
    )
    assert reviewed.review.status == "reviewed"
    assert reviewed.review.decision == "needs_revision"
    assert reviewed.allowed_action == "draft_only"


def test_conflict_resolution_requires_notes_and_known_ids():
    package = evidence_package()
    package = package.model_copy(update={
        "citations": [package.citations[0], package.citations[0].model_copy(update={"citation_id": "E2"})]
    })
    draft = annotate_answer_draft_conflict(
        build_answer_draft(package, ["Claim [E1] [E2]"]), package,
        citation_ids=["E1", "E2"], description="Conflict",
    )
    with pytest.raises(ValueError, match="resolving conflicts requires review notes"):
        record_answer_draft_review(
            draft, package, reviewer="alice", decision="accepted", resolved_conflict_ids=["CF1"]
        )
    with pytest.raises(ValueError, match="unknown conflict"):
        record_answer_draft_review(
            draft, package, reviewer="alice", decision="accepted", notes="Resolved", resolved_conflict_ids=["CF9"]
        )
