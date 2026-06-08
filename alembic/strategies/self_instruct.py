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
        self._multi_turn = bool(params.get("multi_turn", False))
        self._seen_instructions: list[str] = []

    def iter_prompts(self):
        suffix = "_mt" if self._multi_turn else ""
        for i in range(self._target_count):
            existing = "\n".join(f"- {inst[:100]}" for inst in self._seen_instructions[-20:]) if self._seen_instructions else "(no existing data yet)"
            builder = PromptBuilder(lang=self._lang)
            builder.from_template(f"self_instruct_system{suffix}.j2")
            builder.from_template(f"self_instruct_user{suffix}.j2", existing_instructions=existing)
            messages = builder.build()
            prompt_id = f"self_instruct:{i}"
            yield (prompt_id, messages)

    def estimated_count(self) -> int:
        return self._target_count

    def generate(self):
        for prompt_id, messages in self.iter_prompts():
            try:
                raw = self._call_api(messages)
                meta = self._build_metadata(prompt_id)
                samples = self._parse(raw, metadata=meta)
                for s in samples:
                    if (s.instruction and s.output) or s.is_multi_turn:
                        if s.instruction:
                            self._seen_instructions.append(s.instruction)
                        elif s.is_multi_turn:
                            first_user = next((m["content"] for m in s.messages if m.get("role") == "user"), "")
                            if first_user:
                                self._seen_instructions.append(first_user)
                        yield s
                    else:
                        logger.warning(f"[SelfInstruct] empty instruction/output for {prompt_id}, skipping")
            except Exception as e:
                logger.error(f"[SelfInstruct] error for {prompt_id}: {e}")

    def _build_metadata(self, prompt_id: str) -> dict:
        return {"strategy": "self_instruct"}
