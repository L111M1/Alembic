import abc
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


def retry_with_backoff(
    fn: Callable[[], T],
    config: Optional[RetryConfig] = None,
    description: str = "",
) -> T:
    """Run ``fn`` retrying on any exception with exponential backoff.

    This is the single retry primitive used by :class:`RetryDecorator`; keeping
    the backoff logic in one place avoids the drift that previously existed
    between the API-layer and strategy-layer retry loops.
    """
    config = config or RetryConfig()
    delay = config.initial_delay
    last_error: Exception = RuntimeError("no attempt made")
    label = description or "Operation"
    for attempt in range(config.max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < config.max_retries:
                logger.warning(
                    f"{label} failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay = min(delay * config.backoff_multiplier, config.max_delay)
    raise RuntimeError(
        f"{label} failed after {config.max_retries + 1} attempts: {last_error}"
    )


class RetryDecorator(BaseAPIClient):
    """Decorator that wraps a :class:`BaseAPIClient` adding retry-with-backoff."""

    def __init__(self, inner: BaseAPIClient, config: Optional[RetryConfig] = None):
        self._inner = inner
        self._config = config or RetryConfig()

    def supports_json_mode(self) -> bool:
        return self._inner.supports_json_mode()

    def call(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, **kwargs
    ) -> str:
        return retry_with_backoff(
            lambda: self._inner.call(
                messages, temperature=temperature, max_tokens=max_tokens, **kwargs
            ),
            self._config,
            "API call",
        )

    def call_json(
        self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, **kwargs
    ) -> dict:
        return retry_with_backoff(
            lambda: self._inner.call_json(
                messages, temperature=temperature, max_tokens=max_tokens, **kwargs
            ),
            self._config,
            "API call_json",
        )


# Backward-compatible alias for the previous private name.
_RetryWrapper = RetryDecorator


def with_retry(client: BaseAPIClient, config: RetryConfig = None) -> BaseAPIClient:
    return RetryDecorator(client, config)


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
