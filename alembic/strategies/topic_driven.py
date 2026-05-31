import logging
from typing import Iterator

from alembic.api.base import BaseAPIClient
from alembic.prompts.builder import PromptBuilder
from alembic.strategies.base import GenerationStrategy

logger = logging.getLogger(__name__)


class TopicDrivenStrategy(GenerationStrategy):
    def __init__(self, api: BaseAPIClient, params: dict):
        super().__init__(api, params)
        self._topics_raw = params.get("topics", [])
        self._samples_per_topic = int(params.get("samples_per_topic", 1))
        self._total_count = int(params.get("total_count", 0))
        self._plan: list[tuple[str, int]] = self._build_plan()

    def _build_plan(self) -> list[tuple[str, int]]:
        if not self._topics_raw:
            return []
        first = self._topics_raw[0]
        if isinstance(first, dict) and "topic" in first:
            return self._build_weighted_plan()
        else:
            return self._build_flat_plan()

    def _build_flat_plan(self) -> list[tuple[str, int]]:
        plan = []
        for topic in self._topics_raw:
            plan.append((str(topic), self._samples_per_topic))
        logger.info(f"TopicDriven (flat): {len(plan)} topics x {self._samples_per_topic}")
        return plan

    def _build_weighted_plan(self) -> list[tuple[str, int]]:
        items: list[tuple[str, float]] = []
        for entry in self._topics_raw:
            topic = entry.get("topic", "")
            weight = float(entry.get("weight", 1.0))
            if topic and weight > 0:
                items.append((topic, weight))
        if not items:
            return []
        total_weight = sum(w for _, w in items)
        target = self._total_count or self._samples_per_topic * len(items)
        plan = []
        allocated = 0
        for i, (topic, w) in enumerate(items):
            if i == len(items) - 1:
                count = target - allocated
            else:
                count = max(1, round(target * w / total_weight))
            plan.append((topic, count))
            allocated += count
        logger.info(
            f"TopicDriven (weighted): target={target}, "
            f"items={[(t, int(w), cnt) for (t, w), (_, cnt) in zip(items, plan)]}"
        )
        return plan

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        for topic, count in self._plan:
            for i in range(count):
                cur_topic = topic
                if count > 1:
                    cur_topic = f"{topic} ({i + 1}/{count}, generate content different from previous)"
                builder = PromptBuilder(lang=self._lang)
                builder.from_template("topic_driven_system.j2")
                builder.from_template("topic_driven_user.j2", topic=cur_topic)
                messages = builder.build()
                prompt_id = f"topic:{topic}:{i}"
                yield (prompt_id, messages)

    def estimated_count(self) -> int:
        return sum(cnt for _, cnt in self._plan)
