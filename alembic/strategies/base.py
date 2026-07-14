import abc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional

from alembic.api.base import BaseAPIClient, RetryConfig, retry_with_backoff
from alembic.core.parser import JSONResponseParser, ResponseParser
from alembic.core.types import GenerationSample

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class GenerationStrategy(abc.ABC):
    """Base for all generation strategies.

    Subclasses define :meth:`iter_prompts` which yields ``(prompt_id, messages)``
    pairs; the base class handles API dispatch, retry, concurrency, and parsing.

    Strategies with a multi-phase workflow (plan → execute) should instead
    extend :class:`MultiStageStrategy`.
    """

    def __init__(self, api: BaseAPIClient, params: dict):
        self._api = api
        self._params = params
        self._name = self.__class__.__name__
        self._lang = params.get("lang", "en")
        self._concurrency = max(1, int(params.get("concurrency", 1)))
        self._json_mode = bool(params.get("json_mode", True))
        self._parser = self._create_parser()

    # ── parser hook ────────────────────────────────────────────────────

    def _create_parser(self) -> ResponseParser:
        """Override to inject a custom response parser."""
        return JSONResponseParser()

    # ── prompt iteration (subclass responsibility) ─────────────────────

    @abc.abstractmethod
    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        """yield (prompt_id, messages)"""

    def _build_metadata(self, prompt_id: str) -> dict:
        """Override to attach metadata (topic, strategy, etc.) to samples."""
        return {}

    # ── generate ───────────────────────────────────────────────────────

    def generate(self) -> Iterator[GenerationSample]:
        if self._concurrency <= 1:
            yield from self._generate_sequential()
        else:
            yield from self._generate_parallel()

    def _generate_sequential(self) -> Iterator[GenerationSample]:
        for prompt_id, messages in self.iter_prompts():
            meta = self._build_metadata(prompt_id)
            samples = self._call_with_retry(messages, meta, prompt_id)
            if samples is None:
                continue
            for s in samples:
                if (s.instruction and s.output) or s.is_multi_turn:
                    yield s
                else:
                    logger.warning(f"[{self._name}] empty instruction/output, skipping")

    def _generate_parallel(self) -> Iterator[GenerationSample]:
        prompts = list(self.iter_prompts())
        if not prompts:
            return

        logger.info(f"[{self._name}] dispatching {len(prompts)} prompts with concurrency={self._concurrency}")
        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            futures = {}
            for prompt_id, messages in prompts:
                meta = self._build_metadata(prompt_id)
                future = executor.submit(self._call_with_retry, messages, meta, prompt_id)
                futures[future] = prompt_id

            for future in as_completed(futures):
                prompt_id = futures[future]
                try:
                    samples = future.result()
                except Exception as e:
                    logger.warning(f"[{self._name}] failed for {prompt_id}: {e}")
                    continue
                if samples is None:
                    continue
                for s in samples:
                    if (s.instruction and s.output) or s.is_multi_turn:
                        yield s
                    else:
                        logger.warning(f"[{self._name}] empty instruction/output, skipping")

    @abc.abstractmethod
    def estimated_count(self) -> int: ...

    # ── API dispatch & parsing ─────────────────────────────────────────

    def _call_api(self, messages: list[dict], use_json_mode: bool = None,
                  temperature: float = None, max_tokens: int = None) -> str:
        if temperature is None:
            temperature = self._params.get("temperature", 0.8)
        if max_tokens is None:
            max_tokens = self._params.get("max_tokens", 2048)
        kwargs = {}
        if use_json_mode is None:
            use_json_mode = self._json_mode
        if use_json_mode and self._api.supports_json_mode():
            kwargs["response_format"] = {"type": "json_object"}
        return self._api.call(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def _call_and_parse(self, messages: list[dict], metadata: dict = None,
                        temperature: float = None, max_tokens: int = None) -> list[GenerationSample]:
        raw = self._call_api(messages, temperature=temperature, max_tokens=max_tokens)
        return self._parse(response_text=raw, metadata=metadata)

    def _call_with_retry(
        self, messages: list[dict], metadata: dict, prompt_id: str
    ) -> Optional[list[GenerationSample]]:
        try:
            return retry_with_backoff(
                lambda: self._call_and_parse(messages, metadata),
                RetryConfig(max_retries=_MAX_RETRIES),
                f"Generate {prompt_id}",
            )
        except RuntimeError as e:
            logger.warning(str(e))
            return None

    # ── backwards-compatible alias so subclasses that override _parse still work ──

    def _parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        return self._parser.parse(response_text=response_text, metadata=metadata)


class MultiStageStrategy(GenerationStrategy):
    """Base for strategies with separate planning and execution phases.

    Instead of overriding :meth:`generate` (which breaks the base class template
    method), subclasses implement:

    * :meth:`_plan_all` — produce a list of plan items
    * :meth:`_execute_all` — consume plan items and yield samples

    Examples: :class:`EvolInstructStrategy` (evolve → answer),
    :class:`TopicDrivenStrategy` (plan topics → generate samples).
    """

    def generate(self) -> Iterator[GenerationSample]:
        items = self._plan_all()
        if not items:
            return
        yield from self._execute_all(items)

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        return iter([])

    @abc.abstractmethod
    def _plan_all(self) -> list:
        """Produce all plan items (e.g. evolved instructions, planned slots)."""

    @abc.abstractmethod
    def _execute_all(self, items: list) -> Iterator[GenerationSample]:
        """Execute every plan item, yielding 0+ samples per item."""
