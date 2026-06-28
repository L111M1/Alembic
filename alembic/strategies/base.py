import abc
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional

from alembic.api.base import BaseAPIClient
from alembic.core.types import GenerationSample

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4  # 1 initial + 3 retries


class GenerationStrategy(abc.ABC):
    def __init__(self, api: BaseAPIClient, params: dict):
        self._api = api
        self._params = params
        self._name = self.__class__.__name__
        self._lang = params.get("lang", "en")
        self._concurrency = max(1, int(params.get("concurrency", 1)))
        self._json_mode = bool(params.get("json_mode", True))

    @abc.abstractmethod
    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        """yield (prompt_id, messages)"""

    def _build_metadata(self, prompt_id: str) -> dict:
        """Override to attach metadata (topic, strategy, etc.) to samples."""
        return {}

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

    def _call_and_parse(self, messages: list[dict], metadata: dict = None) -> list[GenerationSample]:
        raw = self._call_api(messages)
        return self._parse(response_text=raw, metadata=metadata)

    def _call_with_retry(
        self, messages: list[dict], metadata: dict, prompt_id: str
    ) -> Optional[list[GenerationSample]]:
        """Single retry entry point shared by the sequential and parallel generators."""
        for attempt in range(1, _MAX_RETRIES + 1):
            if attempt > 1:
                logger.info(f"[{self._name}] retry {attempt}/{_MAX_RETRIES} for {prompt_id}")
            try:
                return self._call_and_parse(messages, metadata)
            except Exception as e:
                if attempt == _MAX_RETRIES:
                    logger.warning(
                        f"[{self._name}] failed for {prompt_id} after {_MAX_RETRIES} attempts: {e}"
                    )
                    return None
                logger.warning(
                    f"[{self._name}] {type(e).__name__} for {prompt_id} (attempt {attempt}): {e}"
                )

    @abc.abstractmethod
    def estimated_count(self) -> int: ...

    def _call_api(self, messages: list[dict], use_json_mode: bool = None) -> str:
        temperature = self._params.get("temperature", 0.8)
        max_tokens = self._params.get("max_tokens", 2048)
        kwargs = {}
        if use_json_mode is None:
            use_json_mode = self._json_mode
        if use_json_mode and self._api.supports_json_mode():
            kwargs["response_format"] = {"type": "json_object"}
        return self._api.call(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def _parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        data = json.loads(text)

        if isinstance(data, list):
            results = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                if "messages" in item:
                    sample = GenerationSample(messages=item["messages"])
                    if metadata:
                        sample.metadata = dict(metadata)
                    results.append(sample)
                else:
                    instruction = item.get("instruction", "").strip()
                    output = item.get("output", "").strip()
                    sample = GenerationSample(instruction=instruction, output=output)
                    if metadata:
                        sample.metadata = dict(metadata)
                    results.append(sample)
            return results

        if "messages" in data:
            messages = data["messages"]
            sample = GenerationSample(messages=messages)
            if metadata:
                sample.metadata = metadata
            return [sample]

        instruction = data.get("instruction", "").strip()
        output = data.get("output", "").strip()
        sample = GenerationSample(instruction=instruction, output=output)
        if metadata:
            sample.metadata = metadata
        return [sample]
