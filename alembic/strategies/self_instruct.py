import logging

from alembic.api.base import BaseAPIClient
from alembic.prompts.builder import PromptBuilder
from alembic.strategies.base import GenerationStrategy

logger = logging.getLogger(__name__)


class SelfInstructStrategy(GenerationStrategy):
    def __init__(self, api: BaseAPIClient, params: dict):
        super().__init__(api, params)
        self._concurrency = 1
        self._target_count = int(params.get("target_count", 10))
        self._seen_instructions: list[str] = []

    def iter_prompts(self):
        for i in range(self._target_count):
            existing = "\n".join(f"- {inst[:100]}" for inst in self._seen_instructions[-20:]) if self._seen_instructions else "(no existing data yet)"
            builder = PromptBuilder(lang=self._lang)
            builder.from_template("self_instruct_system.j2")
            builder.from_template("self_instruct_user.j2", existing_instructions=existing)
            messages = builder.build()
            prompt_id = f"self_instruct:{i}"
            yield (prompt_id, messages)

    def estimated_count(self) -> int:
        return self._target_count

    def generate(self):
        for prompt_id, messages in self.iter_prompts():
            try:
                raw = self._call_api(messages)
                samples = self._parse(raw)
                for s in samples:
                    if s.instruction and s.output:
                        self._seen_instructions.append(s.instruction)
                        yield s
                    else:
                        logger.warning(f"[SelfInstruct] empty instruction/output for {prompt_id}, skipping")
            except Exception as e:
                logger.error(f"[SelfInstruct] error for {prompt_id}: {e}")
