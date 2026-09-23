"""LLM configuration for the optional LangGraph binding.

The configuration file only names the environment variable that holds the API
key; the key itself is never stored in the file, committed, or logged. The model
string targets an OpenAI-compatible endpoint (``base_url`` + ``api_key``), which
covers hosted and local gateways alike.
"""

from __future__ import annotations

import os
from pathlib import Path

from typing import Literal

from pydantic import Field, model_validator

from magnetic_vasp_agent.agent.tool_models import StrictModel

WireApi = Literal["chat", "responses"]
ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class LlmConfig(StrictModel):
    """Declarative, secret-free LLM configuration."""

    schema_version: int = Field(default=1, ge=1, le=1)
    provider: str = Field(default="openai_compatible", min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(default="MAGVASP_LLM_API_KEY", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_tool_calls: int = Field(default=12, ge=1, le=100)
    wire_api: WireApi = "chat"
    reasoning_effort: ReasoningEffort | None = None
    disable_response_storage: bool = False

    @model_validator(mode="after")
    def validate_config(self) -> LlmConfig:
        if self.provider != "openai_compatible":
            raise ValueError("only the openai_compatible provider is supported")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) endpoint")
        if any(ch.isspace() for ch in self.api_key_env):
            raise ValueError("api_key_env must be an environment variable name")
        return self

    def resolve_api_key(self) -> str:
        """Read the API key from the environment; never from the config file."""

        key = os.environ.get(self.api_key_env)
        if not key or not key.strip():
            raise RuntimeError(
                f"environment variable {self.api_key_env} is not set; "
                "export the API key before running the LLM path"
            )
        return key


def load_llm_config(path: Path) -> LlmConfig:
    return LlmConfig.model_validate_json(path.read_bytes())
