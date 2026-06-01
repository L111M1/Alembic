import logging
from typing import Any, Iterator

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
        self._plan: list[dict[str, Any]] = self._build_plan()

    def _build_plan(self) -> list[dict[str, Any]]:
        if not self._topics_raw:
            return []
        first = self._topics_raw[0]
        if isinstance(first, dict) and "topic" in first:
            return self._build_weighted_plan()
        else:
            return self._build_flat_plan()

    def _build_flat_plan(self) -> list[dict[str, Any]]:
        plan = []
        for topic in self._topics_raw:
            plan.append({"topic": str(topic), "count": self._samples_per_topic, "knowledge": ""})
        logger.info(f"TopicDriven (flat): {len(plan)} topics x {self._samples_per_topic}")
        return plan

    def _build_weighted_plan(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for entry in self._topics_raw:
            topic = entry.get("topic", "")
            weight = float(entry.get("weight", 1.0))
            if topic and weight > 0:
                items.append({
                    "topic": topic,
                    "weight": weight,
                    "knowledge": entry.get("knowledge", ""),
                })
        if not items:
            return []
        total_weight = sum(it["weight"] for it in items)
        target = self._total_count or self._samples_per_topic * len(items)
        plan = []
        allocated = 0
        for i, it in enumerate(items):
            if i == len(items) - 1:
                count = target - allocated
            else:
                count = max(1, round(target * it["weight"] / total_weight))
            plan.append({"topic": it["topic"], "count": count, "knowledge": it["knowledge"]})
            allocated += count
        logger.info(
            f"TopicDriven (weighted): target={target}, "
            f"items={[(it['topic'], it['weight'], it['count']) for it in plan if 'weight' in it]}"
        )
        return plan

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        for entry in self._plan:
            topic = entry["topic"]
            knowledge = entry.get("knowledge", "")
            for i in range(entry["count"]):
                builder = PromptBuilder(lang=self._lang)
                builder.from_template("topic_driven_system.j2")
                builder.from_template("topic_driven_user.j2", topic=topic, knowledge=knowledge)
                messages = builder.build()
                prompt_id = f"topic:{topic}:{i}"
                yield (prompt_id, messages)

    def _build_metadata(self, prompt_id: str) -> dict:
        parts = prompt_id.split(":")
        return {"strategy": "topic_driven", "topic": parts[1] if len(parts) >= 2 else ""}

    def estimated_count(self) -> int:
        return sum(entry["count"] for entry in self._plan)
