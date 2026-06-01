import logging
import random
from typing import Iterator

from alembic.api.base import BaseAPIClient
from alembic.core.types import SeedSample
from alembic.prompts.builder import PromptBuilder, load_seeds
from alembic.strategies.base import GenerationStrategy

logger = logging.getLogger(__name__)


class SeedDrivenStrategy(GenerationStrategy):
    def __init__(self, api: BaseAPIClient, params: dict):
        super().__init__(api, params)
        self._seeds: list[SeedSample] = []
        seed_file = params.get("seed_file")
        field_map = params.get("field_map")
        if seed_file:
            self._seeds = load_seeds(seed_file, field_map)
        self._example_num = max(1, min(int(params.get("example_num", 3)), len(self._seeds)))
        self._target_count = int(params.get("target_count", 10))

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        if not self._seeds or self._example_num == 0:
            logger.warning("No seeds loaded, skipping SeedDrivenStrategy")
            return
        for i in range(self._target_count):
            chosen = random.sample(self._seeds, min(self._example_num, len(self._seeds)))
            examples_text_parts = []
            for j, seed in enumerate(chosen, 1):
                examples_text_parts.append(f"Example {j}:\n  instruction: {seed.instruction}\n  output: {seed.output}")
            examples_text = "\n\n".join(examples_text_parts)
            builder = PromptBuilder(lang=self._lang)
            builder.from_template("seed_system.j2")
            builder.from_template("seed_user.j2", examples=examples_text)
            messages = builder.build()
            prompt_id = f"seed:{i}"
            yield (prompt_id, messages)

    def _build_metadata(self, prompt_id: str) -> dict:
        return {"strategy": "seed_driven"}

    def estimated_count(self) -> int:
        return self._target_count
