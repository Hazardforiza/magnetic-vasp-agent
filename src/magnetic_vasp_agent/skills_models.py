"""Typed catalog of project skills the agent may read, never execute."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from magnetic_vasp_agent.knowledge.models import StrictModel

SKILL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
ALLOWED_SKILL_SUFFIXES = {".md", ".txt"}
BLOCKED_SKILL_PARTS = {"scripts", ".git", ".system", "__pycache__"}


class SkillRecord(StrictModel):
    id: str
    title: str = Field(min_length=1, max_length=200)
    path: str
    enabled: bool = True
    scripts_allowed: bool = False
    domains: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_record(self) -> SkillRecord:
        if not SKILL_ID_PATTERN.fullmatch(self.id):
            raise ValueError("skill id must be lowercase ASCII kebab-case")
        if self.scripts_allowed:
            raise ValueError("skill scripts must stay disabled for the agent")
        if "\\" in self.path or self.path.strip() != self.path or not self.path:
            raise ValueError("skill path must be a trimmed POSIX-relative path")
        if any(part in {"", ".", ".."} for part in self.path.split("/")):
            raise ValueError("skill path must stay a normalized relative path")
        normalized = [item.casefold() for item in self.domains]
        if len(set(normalized)) != len(normalized):
            raise ValueError("skill domains must be unique")
        return self


class SkillCatalog(StrictModel):
    schema_version: Literal[1]
    base_directory: str = "../.."
    skills: list[SkillRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> SkillCatalog:
        ids = [record.id for record in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("skill ids must be unique")
        if not self.base_directory or self.base_directory.strip() != self.base_directory:
            raise ValueError("base_directory must be non-empty and trimmed")
        if "\\" in self.base_directory:
            raise ValueError("base_directory must use forward slashes")
        return self


class SkillSummary(StrictModel):
    id: str
    title: str
    description: str
    domains: list[str]
    enabled: bool
    scripts_allowed: bool
    reference_files: list[str] = Field(default_factory=list)
    notes: str | None = None


class SkillDocument(StrictModel):
    id: str
    title: str
    description: str
    domains: list[str]
    relative_path: str
    text: str
    scripts_allowed: bool = False
