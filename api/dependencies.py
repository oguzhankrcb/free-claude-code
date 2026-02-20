"""Dependency injection for FastAPI."""

from fastapi import Depends, HTTPException
from loguru import logger

from config.settings import NVIDIA_NIM_BASE_URL, Settings
from config.settings import get_settings as _get_settings
from providers.base import BaseProvider, ProviderConfig



def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def _build_provider(provider_type: str, settings: Settings) -> BaseProvider:
    """Builds and returns a specific provider instance."""
    if provider_type == "nvidia_nim":
        if (
            not settings.nvidia_nim_api_key
            or not settings.nvidia_nim_api_key.strip()
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "NVIDIA_NIM_API_KEY is not set. Add it to your .env file. "
                    "Get a key at https://build.nvidia.com/settings/api-keys"
                ),
            )
        from providers.nvidia_nim import NvidiaNimProvider

        config = ProviderConfig(
            api_key=settings.nvidia_nim_api_key,
            base_url=NVIDIA_NIM_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        provider = NvidiaNimProvider(config, nim_settings=settings.nim)
        logger.info("Provider initialized: %s", provider_type)
        return provider

    elif provider_type == "open_router":
        if (
            not settings.open_router_api_key
            or not settings.open_router_api_key.strip()
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "OPENROUTER_API_KEY is not set. Add it to your .env file. "
                    "Get a key at https://openrouter.ai/keys"
                ),
            )
        from providers.open_router import OpenRouterProvider

        config = ProviderConfig(
            api_key=settings.open_router_api_key,
            base_url="https://openrouter.ai/api/v1",
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        provider = OpenRouterProvider(config)
        logger.info("Provider initialized: %s", provider_type)
        return provider

    elif provider_type == "lmstudio":
        from providers.lmstudio import LMStudioProvider

        config = ProviderConfig(
            api_key="lm-studio",
            base_url=settings.lm_studio_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        provider = LMStudioProvider(config)
        logger.info("Provider initialized: %s", provider_type)
        return provider

    elif provider_type == "vertex_ai":
        if (
            not settings.vertex_ai_api_key
            or not settings.vertex_ai_base_url
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "VERTEX_AI_API_KEY or VERTEX_AI_BASE_URL is not set. Add them to your .env file."
                ),
            )
        from providers.vertex_ai import VertexAIProvider

        config = ProviderConfig(
            api_key=settings.vertex_ai_api_key,
            base_url=settings.vertex_ai_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        provider = VertexAIProvider(config)
        logger.info("Provider initialized: %s", provider_type)
        return provider

    else:
        logger.error(
            "Unknown provider_type: '%s'. Supported: 'nvidia_nim', 'open_router', 'lmstudio', 'vertex_ai'",
            provider_type,
        )
        raise ValueError(
            f"Unknown provider_type: '{provider_type}'. "
            f"Supported: 'nvidia_nim', 'open_router', 'lmstudio', 'vertex_ai'"
        )

# Cache dictionary mapping provider type string to provider instance
_provider_cache: dict[str, BaseProvider] = {}

class ProviderFactory:
    """Factory dependency that resolves providers lazily."""
    def __init__(self, settings: Settings):
        self.settings = settings

    def get(self, provider_type: str | None = None) -> BaseProvider:
        """Get or create the provider instance."""
        target_type = provider_type or self.settings.provider_type
        if target_type not in _provider_cache:
            _provider_cache[target_type] = _build_provider(target_type, self.settings)
        return _provider_cache[target_type]

def get_provider_factory(settings: Settings = Depends(get_settings)) -> ProviderFactory:
    """Dependency injecting the ProviderFactory."""
    return ProviderFactory(settings)

def get_provider() -> BaseProvider:
    """Legacy get_provider (mostly used by tests/routers without DI properly mapping factory).
    It will just return the default provider according to settings."""
    settings = get_settings()
    factory = ProviderFactory(settings)
    return factory.get(settings.provider_type)

async def cleanup_provider():
    """Cleanup all cached provider resources."""
    global _provider_cache
    for provider in _provider_cache.values():
        client = getattr(provider, "_client", None)
        if client and hasattr(client, "aclose"):
            await client.aclose()
    _provider_cache.clear()
    logger.debug("Provider cache cleanup completed")
