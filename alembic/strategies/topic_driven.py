import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator, Optional

from alembic.api.base import BaseAPIClient
from alembic.core.types import GenerationSample
from alembic.prompts.builder import PromptBuilder
from alembic.strategies.base import GenerationStrategy

logger = logging.getLogger(__name__)


class TopicDrivenStrategy(GenerationStrategy):
    def __init__(self, api: BaseAPIClient, params: dict):
        super().__init__(api, params)
        self._topics_raw = params.get("topics", [])
        self._samples_per_topic = int(params.get("samples_per_topic", 1))
        self._total_count = int(params.get("total_count", 0))
        self._multi_turn = bool(params.get("multi_turn", False))
        self._max_samples_per_request = int(params.get("max_samples_per_request", 10))
        self._execution_max_per_request = int(params.get("execution_max_per_request", 2))
        self._two_stage = bool(params.get("two_stage", True))
        self._plan: list[dict[str, Any]] = self._build_plan()
        self._plan_items: Optional[list[dict[str, Any]]] = None  # cached stage-1 output
        self._plan_lookup: dict[str, list[dict[str, Any]]] = {}  # prompt_id -> batch plan items

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

    # ── iter_prompts dispatch ──────────────────────────────────────────

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        if self._two_stage:
            yield from self._iter_prompts_two_stage()
        else:
            yield from self._iter_prompts_one_stage()

    # ── one-stage (original behaviour) ─────────────────────────────────

    def _iter_prompts_one_stage(self) -> Iterator[tuple[str, list[dict]]]:
        suffix = "_mt" if self._multi_turn else ""
        max_batch = self._max_samples_per_request
        for entry in self._plan:
            topic = entry["topic"]
            knowledge = entry.get("knowledge", "")
            total = entry["count"]
            remaining = total
            batch_idx = 0
            while remaining > 0:
                batch_count = min(remaining, max_batch)
                remaining -= batch_count
                builder = PromptBuilder(lang=self._lang)
                builder.from_template(f"topic_driven_system{suffix}.j2")
                builder.from_template(
                    f"topic_driven_user{suffix}.j2",
                    topic=topic, knowledge=knowledge, count=batch_count,
                )
                messages = builder.build()
                prompt_id = (
                    f"topic:{topic}:batch{batch_idx}"
                    if total > max_batch
                    else f"topic:{topic}"
                )
                yield (prompt_id, messages)
                batch_idx += 1

    # ── two-stage: plan → execute ──────────────────────────────────────

    def _iter_prompts_two_stage(self) -> Iterator[tuple[str, list[dict]]]:
        if self._plan_items is None:
            self._plan_items = self._run_planning()

        if not self._plan_items:
            logger.warning("Planning produced no items, nothing to execute")
            return

        suffix = "_mt" if self._multi_turn else ""

        by_topic: dict[str, list[dict[str, Any]]] = {}
        for item in self._plan_items:
            t = item.get("topic", "")
            by_topic.setdefault(t, []).append(item)

        batch_idx = 0
        for topic, topic_items in by_topic.items():
            knowledge = topic_items[0].get("_topic_knowledge", "") if topic_items else ""

            exec_max = max(1, self._execution_max_per_request)
            for chunk_start in range(0, len(topic_items), exec_max):
                chunk = topic_items[chunk_start:chunk_start + exec_max]

                plan_lines = self._format_plan_batch(chunk)
                sub_topic_list = ", ".join(
                    item.get("sub_topic", "") for item in chunk if item.get("sub_topic")
                )

                builder = PromptBuilder(lang=self._lang)
                builder.from_template(f"topic_driven_system{suffix}.j2")
                builder.from_template(
                    f"topic_driven_user{suffix}.j2",
                    topic=topic,
                    knowledge=knowledge,
                    count=len(chunk),
                )
                messages = builder.build()

                plan_header = (
                    f"\n\n--- PLAN ---\n"
                    f"Sub-topics: {sub_topic_list}\n\n"
                    f"MUST generate exactly {len(chunk)} samples following these specifications "
                    f"(one per item, same order):\n"
                    f"{plan_lines}\n\n"
                    f"CRITICAL: Every sample must differ substantially in instruction, output, and structure. No two samples may resemble each other."
                    f"\n--- END PLAN ---"
                )
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += plan_header

                prompt_id = f"topic:{topic}" if len(topic_items) <= exec_max else f"topic:{topic}:batch{batch_idx}"
                self._plan_lookup[prompt_id] = chunk
                yield (prompt_id, messages)
                batch_idx += 1

    def _format_plan_batch(self, batch: list[dict[str, Any]]) -> str:
        lines = []
        for i, item in enumerate(batch):
            lines.append(
                f"  {i + 1}. sub_topic={item.get('sub_topic', '')} | "
                f"angle={item.get('angle', '')} | "
                f"difficulty={item.get('difficulty', 'intermediate')} | "
                f"question_type={item.get('question_type', 'concept_explanation')}"
            )
        return "\n".join(lines)

    # ── stage 1: planning ──────────────────────────────────────────────

    def _run_planning(self) -> list[dict[str, Any]]:
        if not self._plan:
            return []

        max_workers = min(self._concurrency, len(self._plan))

        if max_workers <= 1:
            return self._run_planning_sequential()

        logger.info(f"Planning {len(self._plan)} topics in parallel (workers={max_workers})")
        all_items: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for entry in self._plan:
                future = executor.submit(
                    self._plan_single_topic,
                    entry["topic"],
                    entry["count"],
                    entry.get("knowledge", ""),
                )
                futures[future] = entry["topic"]

            for future in as_completed(futures):
                topic = futures[future]
                try:
                    topic_items = future.result()
                    all_items.extend(topic_items)
                except Exception as e:
                    logger.error(f"Planning failed for topic '{topic}': {e}")

        deduped: list[dict[str, Any]] = []
        seen_angles: set[str] = set()
        for item in all_items:
            angle_key = self._normalize_angle(item.get("angle", ""))
            if angle_key and angle_key not in seen_angles:
                seen_angles.add(angle_key)
                deduped.append(item)
            else:
                logger.debug(
                    f"Skipping duplicate angle (post-dedup): {item.get('angle', '')[:80]}"
                )

        logger.info(
            f"Planning complete: {len(deduped)} unique items "
            f"across {len(self._plan)} topics "
            f"(filtered {len(all_items) - len(deduped)} cross-topic duplicates)"
        )
        self._log_plan_breakdown(deduped)
        return deduped

    def _run_planning_sequential(self) -> list[dict[str, Any]]:
        all_items: list[dict[str, Any]] = []
        seen_angles: set[str] = set()

        for entry in self._plan:
            topic = entry["topic"]
            knowledge = entry.get("knowledge", "")
            remaining = entry["count"]

            while remaining > 0:
                batch_size = min(remaining, self._max_samples_per_request)
                topic_items = self._plan_topic(
                    topic, batch_size, knowledge, list(seen_angles),
                )
                for item in topic_items:
                    item["topic"] = topic
                    item["_topic_knowledge"] = knowledge
                    angle_key = self._normalize_angle(item.get("angle", ""))
                    if angle_key and angle_key not in seen_angles:
                        seen_angles.add(angle_key)
                        all_items.append(item)
                    else:
                        logger.debug(
                            f"Skipping duplicate angle: {item.get('angle', '')[:80]}"
                        )
                remaining -= batch_size

        logger.info(
            f"Planning complete: {len(all_items)} unique items "
            f"across {len(self._plan)} topics "
            f"(filtered {sum(e['count'] for e in self._plan) - len(all_items)} duplicates)"
        )
        self._log_plan_breakdown(all_items)
        return all_items

    def _plan_single_topic(
        self, topic: str, count: int, knowledge: str,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen_angles: set[str] = set()
        remaining = count

        while remaining > 0:
            batch_size = min(remaining, self._max_samples_per_request)
            topic_items = self._plan_topic(
                topic, batch_size, knowledge, list(seen_angles),
            )
            for item in topic_items:
                item["topic"] = topic
                item["_topic_knowledge"] = knowledge
                angle_key = self._normalize_angle(item.get("angle", ""))
                if angle_key and angle_key not in seen_angles:
                    seen_angles.add(angle_key)
                    items.append(item)
                else:
                    logger.debug(
                        f"Skipping duplicate angle: {item.get('angle', '')[:80]}"
                    )
            remaining -= batch_size

        return items

    def _log_plan_breakdown(self, items: list[dict[str, Any]]) -> None:
        by_topic: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            t = item.get("topic", "(unknown)")
            by_topic.setdefault(t, []).append(item)

        for topic, topic_items in sorted(by_topic.items()):
            sub_topic_counts: dict[str, int] = {}
            for item in topic_items:
                st = item.get("sub_topic", "") or "(unnamed)"
                sub_topic_counts[st] = sub_topic_counts.get(st, 0) + 1

            parts = [f"{st}({cnt})" for st, cnt in sorted(sub_topic_counts.items(), key=lambda x: -x[1])]
            logger.info(f"  [{topic}] {len(topic_items)} items: {', '.join(parts[:8])}{'...' if len(parts) > 8 else ''}")

    def _plan_topic(
        self, topic: str, count: int, knowledge: str, existing_angles: list[str],
    ) -> list[dict[str, Any]]:
        builder = PromptBuilder(lang=self._lang)
        builder.from_template("planner_system.j2")

        angle_hint = ""
        if existing_angles:
            recent = existing_angles[-30:]
            angle_hint = "Already planned angles across all topics — DO NOT reuse any of these:\n"
            angle_hint += "\n".join(f"  - {a}" for a in recent)

        builder.from_template(
            "planner_user.j2",
            topic=topic,
            count=count,
            knowledge=knowledge,
            existing_angles=angle_hint,
        )
        messages = builder.build()
        raw = self._call_api(messages, use_json_mode=False)
        items = self._parse_plan_items(raw, topic)
        logger.info(
            f"Planning topic '{topic}': requested {count}, got {len(items)} items"
        )
        return items

    def _parse_plan_items(
        self, response_text: str, topic: str,
    ) -> list[dict[str, Any]]:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse plan JSON for topic '{topic}'")
            return []

        result: list[dict[str, Any]] = []
        entries: list[dict] = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict) and "items" in data:
            entries = data["items"]
        elif isinstance(data, dict) and "plan" in data:
            entries = data["plan"]

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            result.append({
                "sub_topic": str(entry.get("sub_topic", "")).strip(),
                "angle": str(entry.get("angle", "")).strip(),
                "difficulty": str(entry.get("difficulty", "intermediate")).strip(),
                "question_type": str(entry.get("question_type", "concept_explanation")).strip(),
            })
        return result

    @staticmethod
    def _normalize_angle(angle: str) -> str:
        return angle.strip().lower()

    # ── metadata / estimate ────────────────────────────────────────────

    def _build_metadata(self, prompt_id: str) -> dict:
        parts = prompt_id.split(":")
        meta = {
            "strategy": "topic_driven",
            "topic": parts[1] if len(parts) >= 2 else "",
        }
        batch = self._plan_lookup.get(prompt_id)
        if batch:
            meta["_plan_items"] = [
                {
                    "sub_topic": item.get("sub_topic", ""),
                    "angle": item.get("angle", ""),
                    "difficulty": item.get("difficulty", ""),
                    "question_type": item.get("question_type", ""),
                }
                for item in batch
            ]
        return meta

    def _parse(self, response_text: str, metadata: dict = None) -> list[GenerationSample]:
        plan_items = None
        if metadata and "_plan_items" in metadata:
            plan_items = metadata.pop("_plan_items")

        samples = super()._parse(response_text, metadata)

        if plan_items:
            for i, sample in enumerate(samples):
                if i < len(plan_items):
                    if sample.metadata is None:
                        sample.metadata = {}
                    sample.metadata["sub_topic"] = plan_items[i]["sub_topic"]
                    sample.metadata["angle"] = plan_items[i]["angle"]
                    sample.metadata["difficulty"] = plan_items[i]["difficulty"]
                    sample.metadata["question_type"] = plan_items[i]["question_type"]

        return samples

    def estimated_count(self) -> int:
        return sum(entry["count"] for entry in self._plan)
