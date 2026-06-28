import logging
from typing import Optional, Type

from alembic.api.base import BaseAPIClient, RetryConfig, with_retry
from alembic.api.providers import OpenAICompatibleClient

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry mapping a provider name to a :class:`BaseAPIClient` subclass.

    New providers can be plugged in without touching the factory:
    ``ProviderRegistry.register("anthropic", AnthropicClient)``.
    """

    def __init__(self):
        self._map: dict[str, Type[BaseAPIClient]] = {}

    def register(self, name: str, cls: Type[BaseAPIClient]) -> None:
        self._map[name] = cls

    def get(self, name: str) -> Optional[Type[BaseAPIClient]]:
        return self._map.get(name)

    def names(self) -> list[str]:
        return list(self._map)


# Default registry with the built-in provider pre-registered.
registry = ProviderRegistry()
registry.register("openai", OpenAICompatibleClient)


def create_client(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    retry: Optional[dict] = None,
    provider: str = "openai",
    **kwargs,
) -> BaseAPIClient:
    cls = registry.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'")
    client = cls(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )

    if retry:
        rc = RetryConfig(
            max_retries=retry.get("max_retries", 3),
            initial_delay=retry.get("initial_delay", 1.0),
            backoff_multiplier=retry.get("backoff_multiplier", 2.0),
            max_delay=retry.get("max_delay", 30.0),
        )
        client = with_retry(client, rc)

    logger.info(f"Created client (provider={provider}, model={model})")
    return client
