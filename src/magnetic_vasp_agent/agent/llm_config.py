"""LLM configuration for the optional LangGraph binding.

The local config may contain a masked optional API key or name an environment
variable that holds it. The model string targets an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, SecretStr, model_validator

from magnetic_vasp_agent.agent.tool_models import StrictModel

WireApi = Literal["chat", "responses"]
ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class LlmConfig(StrictModel):
    """Local LLM config with an optional masked API key."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    provider: str = Field(default="openai_compatible", min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(
        default="MAGVASP_LLM_API_KEY",
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    api_key: SecretStr | None = Field(default=None, repr=False)
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
        return self

    def resolve_api_key(self) -> str:
        """Read a local inline key first, then fall back to the configured env var."""

        if self.api_key is not None:
            key = self.api_key.get_secret_value()
            if key.strip():
                return key
        key = os.environ.get(self.api_key_env)
        if not key or not key.strip():
            raise RuntimeError(
                "no API key is configured; enter api_key in the local llm.json "
                "or set the environment variable named by api_key_env."
            )
        return key


def load_llm_config(path: Path) -> LlmConfig:
    return LlmConfig.model_validate_json(path.read_bytes())
