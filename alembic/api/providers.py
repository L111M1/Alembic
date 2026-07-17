import logging
import os
from typing import Optional

import httpx
from openai import OpenAI

from alembic.api.base import BaseAPIClient

logger = logging.getLogger(__name__)


class OpenAICompatibleClient(BaseAPIClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self._model = model
        key = api_key or os.environ.get("API_KEY", "")
        url = base_url or os.environ.get("BASE_URL", "") or None
        # Use a connection pool with short keep-alive expiry.
        # Cloudflare fronted endpoints (e.g. inferknock.ai) close idle
        # connections after ~60s; setting keepalive_expiry below that
        # ensures we proactively retire stale connections before the server
        # does, avoiding "[SSL: UNEXPECTED_EOF_WHILE_READING]" errors.
        self._http_client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=128,
                max_keepalive_connections=32,
                keepalive_expiry=30.0,
            ),
        )
        self._client = OpenAI(
            api_key=key,
            base_url=url,
            timeout=timeout,
            http_client=self._http_client,
        )

    def supports_json_mode(self) -> bool:
        return True

    def call(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> str:
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        params.update(kwargs)
        response = self._client.chat.completions.create(**params)
        return response.choices[0].message.content or ""
