import pytest
from pydantic import ValidationError

from magnetic_vasp_agent.knowledge.retrieval import (
    build_answer_draft,
    evidence_package_sha256,
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
