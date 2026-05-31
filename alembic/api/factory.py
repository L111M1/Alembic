import logging
from typing import Optional

from alembic.api.base import BaseAPIClient, RetryConfig, with_retry
from alembic.api.providers import OpenAICompatibleClient

logger = logging.getLogger(__name__)


def create_client(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    retry: Optional[dict] = None,
    **kwargs,
) -> BaseAPIClient:
    client = OpenAICompatibleClient(
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

    logger.info(f"Created client (model={model})")
    return client
