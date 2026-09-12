from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from magnetic_vasp_agent.knowledge.models import (
    CalculationTask,
    KnowledgeSourceType,
    MagneticMode,
    StrictModel,
)


CITATION_MARKER_PATTERN = re.compile(r"\[(E[1-9][0-9]*)\]")


class RetrievalDocument(StrictModel):
    schema_version: Literal[1]
    document_id: str
    source_id: str
    source_type: KnowledgeSourceType
    authority: int = Field(ge=0, le=100)
    source_validated: bool
    title: str
    source_url: str | None = None
    local_relative_path: str | None = None
    page_id: int | None = Field(default=None, gt=0)
    revision_id: int | None = Field(default=None, gt=0)
    section_path: list[str]
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tasks: list[CalculationTask]
    magnetic_modes: list[MagneticMode]


class RetrievalIndexManifest(StrictModel):
    schema_version: Literal[1]
    index_kind: Literal["bm25"]
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    tokenizer_version: str
    document_count: int = Field(gt=0)
    documents_relative_path: str
    documents_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_types: list[KnowledgeSourceType]


class VectorIndexManifest(StrictModel):
    schema_version: Literal[1]
    index_kind: Literal["qdrant"]
    corpus_index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents_relative_path: str
    documents_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qdrant_relative_path: str
    collection_name: str = Field(min_length=1, max_length=100)
    embedding_provider: Literal["hash", "fastembed"]
    embedding_model: str = Field(min_length=1, max_length=200)
    embedding_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    embedding_runtime: str | None = None
    vector_dimension: int = Field(gt=0, le=4096)
    reranker: Literal["rrf-v1"]
    document_count: int = Field(gt=0)
    source_types: list[KnowledgeSourceType]

    @model_validator(mode="after")
    def embedding_contract(self) -> VectorIndexManifest:
        if self.embedding_provider == "hash":
            if (self.embedding_model != "hash-token-bigram-v1" or self.vector_dimension != 384
                    or self.embedding_fingerprint is not None or self.embedding_runtime is not None):
                raise ValueError("unsupported hash embedding contract")
        else:
            if (self.embedding_model != "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    or self.vector_dimension != 384 or not self.embedding_fingerprint
                    or not self.embedding_runtime):
                raise ValueError("local embedding contract is incomplete or unsupported")
        return self


class RetrievalFilter(StrictModel):
    source_types: list[KnowledgeSourceType] = Field(default_factory=list)
    tasks: list[CalculationTask] = Field(default_factory=list)
    magnetic_modes: list[MagneticMode] = Field(default_factory=list)
    validated_only: bool = False


class Evidence(StrictModel):
    rank: int = Field(gt=0)
    score: float = Field(ge=0)
    document_id: str
    source_id: str
    source_type: KnowledgeSourceType
    authority: int = Field(ge=0, le=100)
    source_validated: bool
    title: str
    source_url: str | None = None
    local_relative_path: str | None = None
    page_id: int | None = Field(default=None, gt=0)
    revision_id: int | None = Field(default=None, gt=0)
    section_path: list[str]
    quote: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class RetrievalResponse(StrictModel):
    schema_version: Literal[1]
    query: str = Field(min_length=1)
    filters: RetrievalFilter
    index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: list[Evidence]
    status: Literal["evidence_found", "evidence_missing"]


class CitedEvidence(Evidence):
    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")


class EvidenceRisk(StrictModel):
    code: Literal[
        "evidence_missing",
        "missing_required_source_type",
        "unvalidated_evidence",
        "human_review_requested",
    ]
    message: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)
    source_types: list[KnowledgeSourceType] = Field(default_factory=list)


class EvidencePackage(StrictModel):
    schema_version: Literal[1]
    question: str = Field(min_length=1)
    retrieval_backend: Literal["bm25", "vector"]
    filters: RetrievalFilter
    index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    citations: list[CitedEvidence]
    required_source_types: list[KnowledgeSourceType]
    source_types_present: list[KnowledgeSourceType]
    missing_source_types: list[KnowledgeSourceType]
    risks: list[EvidenceRisk]
    status: Literal["evidence_missing", "review_required", "evidence_ready"]
    allowed_action: Literal["draft_only"] = "draft_only"
    human_review_required: bool
    conflict_assessment: Literal["not_semantically_assessed"] = (
        "not_semantically_assessed"
    )
    answering_rules: list[str]


class AnswerDraftClaim(StrictModel):
    """One technical assertion in an answer draft.

    ``citation_ids`` is derived from the literal ``[E#]`` markers in ``text``.
    Keeping it explicitly in the serialized draft makes the cited assertions easy
    to review without reparsing prose, while the validator prevents the two
    representations from drifting apart.
    """

    claim_id: str = Field(pattern=r"^C[1-9][0-9]*$")
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def citation_ids_match_text(self) -> AnswerDraftClaim:
        markers = list(dict.fromkeys(CITATION_MARKER_PATTERN.findall(self.text)))
        if self.citation_ids != markers:
            raise ValueError(
                "claim citation_ids must exactly match [E#] markers in claim text"
            )
        return self


class CitationIntegrity(StrictModel):
    """Machine-checkable result of comparing a draft with its evidence package."""

    available_citation_ids: list[str]
    cited_citation_ids: list[str]
    unknown_citation_ids: list[str]
    malformed_citation_markers: list[str]
    uncited_claim_ids: list[str]
    citation_complete: bool

    @model_validator(mode="after")
    def complete_matches_findings(self) -> CitationIntegrity:
        for field_name in (
            "available_citation_ids",
            "cited_citation_ids",
            "unknown_citation_ids",
            "malformed_citation_markers",
            "uncited_claim_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        expected_complete = not (
            self.unknown_citation_ids
            or self.malformed_citation_markers
            or self.uncited_claim_ids
        )
        if self.citation_complete != expected_complete:
            raise ValueError("citation_complete does not match citation findings")
        return self


class AnswerDraftReviewReason(StrictModel):
    code: Literal[
        "human_review_required",
        "citation_incomplete",
        "evidence_missing",
        "evidence_package_review_required",
        "evidence_conflict",
    ]
    message: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)


class EvidenceConflict(StrictModel):
    """A reviewer supplied annotation that two or more citations disagree."""

    conflict_id: str = Field(pattern=r"^CF[1-9][0-9]*$")
    citation_ids: list[str] = Field(min_length=2)
    description: str = Field(min_length=1)
    status: Literal["unresolved", "resolved"] = "unresolved"
    resolution: str | None = None

    @model_validator(mode="after")
    def conflict_contract(self) -> EvidenceConflict:
        if len(self.citation_ids) != len(set(self.citation_ids)):
            raise ValueError("conflict citation_ids must be unique")
        if any(not re.fullmatch(r"E[1-9][0-9]*", item) for item in self.citation_ids):
            raise ValueError("conflict citation_ids must use E# identifiers")
        if self.status == "resolved" and not (self.resolution and self.resolution.strip()):
            raise ValueError("resolved conflicts require a resolution")
        if self.status == "unresolved" and self.resolution is not None:
            raise ValueError("unresolved conflicts cannot include a resolution")
        return self


class AnswerDraftReview(StrictModel):
    """Human review record; it never grants permission to execute a draft."""

    status: Literal["review_required", "reviewed"] = "review_required"
    reviewer: str | None = None
    decision: Literal["pending", "accepted", "rejected", "needs_revision"] = "pending"
    notes: str | None = None
    resolved_conflict_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def review_contract(self) -> AnswerDraftReview:
        if len(self.resolved_conflict_ids) != len(set(self.resolved_conflict_ids)):
            raise ValueError("resolved_conflict_ids must be unique")
        if self.status == "reviewed":
            if not (self.reviewer and self.reviewer.strip()):
                raise ValueError("reviewed drafts require reviewer")
            if self.decision == "pending":
                raise ValueError("reviewed drafts require a decision")
        elif self.decision != "pending":
            raise ValueError("review_required drafts must have pending decision")
        return self


class AnswerDraft(StrictModel):
    """A non-executable response draft bound to one immutable evidence package."""

    schema_version: Literal[1]
    question: str = Field(min_length=1)
    evidence_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_package_status: Literal[
        "evidence_missing", "review_required", "evidence_ready"
    ]
    claims: list[AnswerDraftClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    citation_integrity: CitationIntegrity
    status: Literal["draft_ready_for_review", "review_required"]
    human_review_required: Literal[True] = True
    allowed_action: Literal["draft_only"] = "draft_only"
    review_reasons: list[AnswerDraftReviewReason] = Field(min_length=1)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    review: AnswerDraftReview = Field(default_factory=AnswerDraftReview)

    @model_validator(mode="after")
    def unique_claim_ids(self) -> AnswerDraft:
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("answer draft claim ids must be unique")
        if any(not item.strip() for item in self.limitations):
            raise ValueError("answer draft limitations must be non-empty")
        cited = list(
            dict.fromkeys(
                citation
                for claim in self.claims
                for citation in claim.citation_ids
            )
        )
        if self.citation_integrity.cited_citation_ids != cited:
            raise ValueError("draft citation integrity does not match claim citations")
        conflict_ids = [item.conflict_id for item in self.conflicts]
        if len(conflict_ids) != len(set(conflict_ids)):
            raise ValueError("answer draft conflict ids must be unique")
        known_conflicts = set(conflict_ids)
        if any(item not in known_conflicts for item in self.review.resolved_conflict_ids):
            raise ValueError("review references an unknown conflict")
        return self


class EvaluationCaseResult(StrictModel):
    case_id: str
    query: str
    result_count: int = Field(ge=0)
    required_source_type_recall: float = Field(ge=0, le=1)
    evidence_terms_found: int = Field(ge=0)
    evidence_terms_total: int = Field(ge=1)
    missing_evidence_terms: list[str]
    forbidden_exact_quotes_found: int = Field(ge=0)
    top_document_ids: list[str]
    status: Literal["retrieved", "no_evidence"]


class EvaluationReport(StrictModel):
    schema_version: Literal[1]
    index_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(gt=0)
    retrieved_cases: int = Field(ge=0)
    retrieval_rate: float = Field(ge=0, le=1)
    required_source_type_coverage: float = Field(ge=0, le=1)
    evidence_term_coverage: float = Field(ge=0, le=1)
    forbidden_exact_quote_rate: float = Field(ge=0, le=1)
    cases: list[EvaluationCaseResult]
