"""Vertex AI provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body


class VertexAIProvider(OpenAICompatibleProvider):
    """Vertex AI provider using OpenAI-compatible OpenAPI endpoint."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="VERTEX",
            base_url=config.base_url,
            api_key="",  # We use default_query instead
            default_query={"key": config.api_key} if config.api_key else None,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
