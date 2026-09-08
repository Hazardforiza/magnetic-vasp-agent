from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from magnetic_vasp_agent.models import CalculationTask, MagneticMode

SOURCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
EVALUATION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,79}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class KnowledgeSourceType(StrEnum):
    OFFICIAL = "official"
    TEAM_POLICY = "team_policy"
    VALIDATED_CASE = "validated_case"
    PERSONAL_NOTE = "personal_note"
    RUNTIME_EXPERIENCE = "runtime_experience"


class KnowledgeLocationType(StrEnum):
    LOCAL_FILE = "local_file"
    LOCAL_DIRECTORY = "local_directory"
    EXTERNAL_URL = "external_url"


class ContentHandling(StrEnum):
    PROJECT = "project"
    LOCAL_ONLY = "local_only"
    REFERENCE_ONLY = "reference_only"


class InventoryStatus(StrEnum):
    INDEXED_METADATA = "indexed_metadata"
    REFERENCE_ONLY = "reference_only"
    MISSING_OPTIONAL = "missing_optional"


AUTHORITY_BY_TYPE: dict[KnowledgeSourceType, int] = {
    KnowledgeSourceType.OFFICIAL: 100,
    KnowledgeSourceType.TEAM_POLICY: 80,
    KnowledgeSourceType.VALIDATED_CASE: 60,
    KnowledgeSourceType.RUNTIME_EXPERIENCE: 50,
    KnowledgeSourceType.PERSONAL_NOTE: 30,
}


class KnowledgeSource(StrictModel):
    id: str
    title: str = Field(min_length=1, max_length=200)
    source_type: KnowledgeSourceType
    location_type: KnowledgeLocationType
    content_handling: ContentHandling
    path: str | None = None
    url: HttpUrl | None = None
    required: bool = True
    validated: bool = False
    vasp_versions: list[str] = Field(default_factory=list)
    tasks: list[CalculationTask] = Field(default_factory=list)
    magnetic_modes: list[MagneticMode] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_source(self) -> KnowledgeSource:
        if not SOURCE_ID_PATTERN.fullmatch(self.id):
            raise ValueError("source id must be lowercase ASCII kebab-case")
        if self.location_type == KnowledgeLocationType.EXTERNAL_URL:
            if self.url is None or self.path is not None:
                raise ValueError("external_url requires url and forbids path")
            if self.content_handling != ContentHandling.REFERENCE_ONLY:
                raise ValueError("external_url must remain reference_only before capture")
            if self.extensions:
                raise ValueError("external_url cannot declare file extensions")
        else:
            if self.path is None or self.url is not None:
                raise ValueError("local source requires path and forbids url")
            if self.content_handling == ContentHandling.REFERENCE_ONLY:
                raise ValueError("local sources cannot be reference_only")
            if self.path.strip() != self.path or not self.path:
                raise ValueError("local source path must be non-empty and trimmed")
        if self.location_type == KnowledgeLocationType.LOCAL_DIRECTORY:
            if not self.extensions:
                raise ValueError("local_directory requires an extension allowlist")
        elif self.extensions:
            raise ValueError("extensions are valid only for local_directory")
        normalized_extensions = []
        for extension in self.extensions:
            normalized = extension.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,10}", normalized):
                raise ValueError(f"unsafe file extension {extension!r}")
            normalized_extensions.append(normalized)
        if len(set(normalized_extensions)) != len(normalized_extensions):
            raise ValueError("extensions must be unique after case normalization")
        if self.source_type == KnowledgeSourceType.VALIDATED_CASE and not self.validated:
            raise ValueError("validated_case must be explicitly validated")
        return self

    @property
    def authority(self) -> int:
        return AUTHORITY_BY_TYPE[self.source_type]


class KnowledgeCatalog(StrictModel):
    schema_version: Literal[1]
    base_directory: str = "../.."
    max_file_bytes: int = Field(default=64 * 1024 * 1024, ge=1, le=512 * 1024 * 1024)
    max_files: int = Field(default=2000, ge=1, le=10000)
    sources: list[KnowledgeSource] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> KnowledgeCatalog:
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("knowledge source ids must be unique")
        if not self.base_directory or self.base_directory.strip() != self.base_directory:
            raise ValueError("base_directory must be non-empty and trimmed")
        return self


class InventoryDocument(StrictModel):
    document_id: str
    relative_path: str
    suffix: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceInventory(StrictModel):
    source_id: str
    title: str
    source_type: KnowledgeSourceType
    authority: int = Field(ge=0, le=100)
    validated: bool
    content_handling: ContentHandling
    vasp_versions: list[str]
    tasks: list[CalculationTask]
    magnetic_modes: list[MagneticMode]
    location: str
    status: InventoryStatus
    declaration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents: list[InventoryDocument] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DuplicateContentGroup(StrictModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_ids: list[str] = Field(min_length=2)


class InventorySummary(StrictModel):
    declared_sources: int = Field(ge=0)
    local_documents: int = Field(ge=0)
    local_bytes: int = Field(ge=0)
    reference_only_sources: int = Field(ge=0)
    missing_optional_sources: int = Field(ge=0)
    duplicate_content_groups: int = Field(ge=0)


class KnowledgeInventory(StrictModel):
    schema_version: Literal[1]
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_directory: str
    sources: list[SourceInventory]
    duplicate_content: list[DuplicateContentGroup]
    summary: InventorySummary


class RagEvaluationCase(StrictModel):
    id: str
    question: str = Field(min_length=5)
    expected_facts: list[str] = Field(min_length=1)
    evidence_terms: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    required_source_types: list[KnowledgeSourceType] = Field(min_length=1)
    task: CalculationTask | None = None
    magnetic_mode: MagneticMode | None = None
    requires_human_review: bool = False

    @model_validator(mode="after")
    def validate_id(self) -> RagEvaluationCase:
        if not EVALUATION_ID_PATTERN.fullmatch(self.id):
            raise ValueError("evaluation id must be lowercase ASCII kebab-case")
        if any(not term.strip() or term != term.strip() for term in self.evidence_terms):
            raise ValueError("evidence_terms must be non-empty and trimmed")
        if len({term.casefold() for term in self.evidence_terms}) != len(
            self.evidence_terms
        ):
            raise ValueError("evidence_terms must be unique ignoring case")
        return self


class RagEvaluationSuite(StrictModel):
    schema_version: Literal[1]
    cases: list[RagEvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self) -> RagEvaluationSuite:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("RAG evaluation case ids must be unique")
        return self
