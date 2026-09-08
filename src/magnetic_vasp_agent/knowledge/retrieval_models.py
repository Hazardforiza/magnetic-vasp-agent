from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from magnetic_vasp_agent.knowledge.models import (
    CalculationTask,
    KnowledgeSourceType,
    MagneticMode,
    StrictModel,
)


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
