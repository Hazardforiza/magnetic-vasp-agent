from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from magnetic_vasp_agent.knowledge.models import (
    KnowledgeSourceType,
    StrictModel,
)
from magnetic_vasp_agent.models import CalculationTask, MagneticMode


class WikiChunk(StrictModel):
    schema_version: Literal[1]
    chunk_id: str
    source_id: str
    source_type: KnowledgeSourceType
    authority: int = Field(ge=0, le=100)
    source_validated: bool
    source_url: str
    page_title: str
    page_id: int = Field(gt=0)
    revision_id: int = Field(gt=0)
    section_path: list[str]
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tasks: list[CalculationTask]
    magnetic_modes: list[MagneticMode]


class WikiSourceCapture(StrictModel):
    source_id: str
    source_url: str
    api_url: str
    page_title: str
    page_id: int = Field(gt=0)
    revision_id: int = Field(gt=0)
    fetched_at: datetime
    raw_relative_path: str
    raw_size_bytes: int = Field(gt=0)
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks_relative_path: str
    chunks_size_bytes: int = Field(gt=0)
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_count: int = Field(gt=0)


class WikiCaptureManifest(StrictModel):
    schema_version: Literal[1]
    capture_kind: Literal["vasp_wiki_mediawiki_api"]
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allowed_hosts: list[str]
    source_count: int = Field(gt=0)
    sources: list[WikiSourceCapture]

    @model_validator(mode="after")
    def validate_sources(self) -> WikiCaptureManifest:
        if self.source_count != len(self.sources):
            raise ValueError("Wiki source_count does not match sources")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Wiki capture source ids must be unique")
        return self
