import abc
import json
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BaseAPIClient(abc.ABC):
    @abc.abstractmethod
    def call(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, **kwargs
    ) -> str:
        """return raw text response from the model"""

    @abc.abstractmethod
    def supports_json_mode(self) -> bool: ...

    def call_json(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, **kwargs
    ) -> dict:
        raw = self.call(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        return _extract_json(raw)


@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 30.0


def with_retry(client: BaseAPIClient, config: RetryConfig = None) -> BaseAPIClient:
    config = config or RetryConfig()
    return _RetryWrapper(client, config)


class _RetryWrapper(BaseAPIClient):
    def __init__(self, inner: BaseAPIClient, config: RetryConfig):
        self._inner = inner
        self._config = config

    def supports_json_mode(self) -> bool:
        return self._inner.supports_json_mode()

    def call(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> str:
        delay = self._config.initial_delay
        last_error = ""
        for attempt in range(self._config.max_retries + 1):
            try:
                return self._inner.call(messages, temperature, max_tokens, **kwargs)
            except Exception as e:
                last_error = str(e)
                if attempt < self._config.max_retries:
                    logger.warning(f"API call failed (attempt {attempt+1}): {last_error}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * self._config.backoff_multiplier, self._config.max_delay)
        raise RuntimeError(f"API call failed after {self._config.max_retries+1} attempts: {last_error}")

    def call_json(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> dict:
        delay = self._config.initial_delay
        last_error = ""
        for attempt in range(self._config.max_retries + 1):
            try:
                return self._inner.call_json(messages, temperature, max_tokens, max_tokens=max_tokens, **kwargs)
            except Exception as e:
                last_error = str(e)
                if attempt < self._config.max_retries:
                    time.sleep(delay)
                    delay = min(delay * self._config.backoff_multiplier, self._config.max_delay)
        raise RuntimeError(f"API call failed after {self._config.max_retries+1} attempts: {last_error}")


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    for char in "{[":
        if char in raw:
            try:
                return json.loads(raw[raw.index(char):raw.rindex(char == "{" and "}" or "]") + 1])
            except (json.JSONDecodeError, ValueError):
                pass
    raise ValueError(f"Could not extract valid JSON from response: {raw[:200]}")
