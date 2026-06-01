import abc
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional

from alembic.api.base import BaseAPIClient
from alembic.core.types import GenerationSample
from alembic.prompts.builder import PromptBuilder

logger = logging.getLogger(__name__)


class GenerationStrategy(abc.ABC):
    def __init__(self, api: BaseAPIClient, params: dict):
        self._api = api
        self._params = params
        self._name = self.__class__.__name__
        self._lang = params.get("lang", "en")
        self._concurrency = max(1, int(params.get("concurrency", 1)))

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
            try:
                raw = self._call_api(messages)
                meta = self._build_metadata(prompt_id)
                samples = self._parse(response_text=raw, metadata=meta)
                for s in samples:
                    if s.instruction and s.output:
                        yield s
                    else:
                        logger.warning(f"[{self._name}] empty instruction/output, skipping")
            except json.JSONDecodeError as e:
                logger.warning(f"[{self._name}] JSON parse error for {prompt_id}: {e}")
            except Exception as e:
                logger.error(f"[{self._name}] generation error for {prompt_id}: {e}")

    def _generate_parallel(self) -> Iterator[GenerationSample]:
        prompts = list(self.iter_prompts())
        if not prompts:
            return

        logger.info(f"[{self._name}] dispatching {len(prompts)} prompts with concurrency={self._concurrency}")
        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            futures = {}
            for prompt_id, messages in prompts:
                meta = self._build_metadata(prompt_id)
                future = executor.submit(self._call_and_parse, messages, meta)
                futures[future] = prompt_id

            for future in as_completed(futures):
                prompt_id = futures[future]
                try:
                    samples = future.result()
                    for s in samples:
                        if s.instruction and s.output:
                            yield s
                        else:
                            logger.warning(f"[{self._name}] empty instruction/output, skipping")
                except json.JSONDecodeError as e:
                    logger.warning(f"[{self._name}] JSON parse error for {prompt_id}: {e}")
                except Exception as e:
                    logger.error(f"[{self._name}] generation error for {prompt_id}: {e}")

    def _call_and_parse(self, messages: list[dict], metadata: dict = None) -> list[GenerationSample]:
        raw = self._call_api(messages)
        return self._parse(response_text=raw, metadata=metadata)

    def estimated_count(self) -> int:
        return len(self._params.get("topics", [])) * self._params.get("samples_per_topic", 1)

    def _call_api(self, messages: list[dict]) -> str:
        temperature = self._params.get("temperature", 0.8)
        max_tokens = self._params.get("max_tokens", 2048)
        return self._api.call(messages, temperature=temperature, max_tokens=max_tokens)

    def _parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        instruction = data.get("instruction", "").strip()
        output = data.get("output", "").strip()
        sample = GenerationSample(instruction=instruction, output=output)
        if metadata:
            sample.metadata = metadata
        return [sample]
