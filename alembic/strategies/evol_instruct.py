import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional

from alembic.api.base import BaseAPIClient, RetryConfig, retry_with_backoff
from alembic.core.types import GenerationSample
from alembic.prompts.builder import PromptBuilder, load_seeds
from alembic.strategies.base import MultiStageStrategy

logger = logging.getLogger(__name__)


class EvolInstructStrategy(MultiStageStrategy):
    """Iterative instruction evolution (Evol-Instruct) strategy.

    Phase 1 — Evolution (``_plan_all``): seeds evolve through N rounds of
    depth/breadth mutations, producing ``(instruction, metadata)`` items.

    Phase 2 — Answer (``_execute_all``): each evolved instruction gets an
    output via a separate API call.

    Compared to the existing seed_driven evolution (one-shot random
    crossover/mutate), this strategy chains mutations iteratively so each
    round builds on the previous.
    """

    _DEFAULT_DEPTH_MUTATIONS = [
        {"name": "add_constraint", "prompt": "Add one or more specific constraints or requirements to the instruction"},
        {"name": "deepen", "prompt": "Increase the depth and breadth of the inquiry"},
        {"name": "concretize", "prompt": "Replace general concepts with more specific concepts"},
        {"name": "increase_reasoning", "prompt": "Request explicit multiple-step reasoning"},
    ]

    _REFUSAL_PHRASES = [
        "sorry", "i cannot", "i can't", "unable to", "as an ai",
        "i am not able", "i'm not able", "not possible",
    ]

    def __init__(self, api: BaseAPIClient, params: dict):
        super().__init__(api, params)
        seed_file = params.get("seed_file")
        field_map = params.get("field_map")
        if not seed_file:
            raise ValueError("evol_instruct requires seed_file")
        seeds = load_seeds(seed_file, field_map)
        self._seeds = [s for s in seeds if s.instruction.strip()]
        if not self._seeds:
            raise ValueError("No seeds with valid instructions found")

        self._max_rounds = max(1, int(params.get("max_rounds", 3)))
        self._branch_factor = max(0, int(params.get("branch_factor", 1)))
        self._depth_rate = self._clamp(params.get("depth_rate", 0.7))
        self._depth_mutations = params.get("depth_mutations") or list(self._DEFAULT_DEPTH_MUTATIONS)
        self._min_ratio = float(params.get("min_evolution_ratio", 0.5))
        self._max_ratio = float(params.get("max_evolution_ratio", 5.0))
        self._generate_output = bool(params.get("generate_output", True))
        self._require_reasoning = bool(params.get("require_reasoning", False))
        self._evol_concurrency = max(1, int(params.get("evol_concurrency", 1)))
        self._evol_temperature = float(params.get("evol_temperature", 0.8))
        self._evol_max_tokens = int(params.get("evol_max_tokens", 1024))
        self._answer_temperature = float(params.get("answer_temperature", 0.6))
        self._answer_max_tokens = int(params.get("answer_max_tokens", 2048))
        self._include_seeds = bool(params.get("include_seeds", False))

        self._evolved_items: list[tuple[str, dict]] = []

    @staticmethod
    def _clamp(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    # ── MultiStageStrategy hooks ───────────────────────────────────

    def _plan_all(self) -> list[tuple[str, dict]]:
        return self._run_evolution()

    def _execute_all(self, items: list[tuple[str, dict]]) -> Iterator[GenerationSample]:
        if self._generate_output:
            yield from self._generate_outputs(items)
        else:
            for inst, meta in items:
                yield GenerationSample(instruction=inst, metadata=meta)

    # ── Phase 1: evolution ──────────────────────────────────────────

    def _run_evolution(self) -> list[tuple[str, dict]]:
        pool: list[tuple[str, dict]] = [
            (s.instruction, {
                "strategy": "evol_instruct",
                "seed_index": i,
                "evolution_round": 0,
                "evolution_type": "seed",
                "evolution_chain": [s.instruction],
            })
            for i, s in enumerate(self._seeds)
        ]
        all_evolved: list[tuple[str, dict]] = []

        for round_num in range(1, self._max_rounds + 1):
            logger.info(
                "Evolution round %d/%d, pool=%d, concurrency=%d",
                round_num, self._max_rounds, len(pool), self._evol_concurrency,
            )
            next_pool = self._run_evolution_round(pool, round_num)
            pool = next_pool
            self._log_pool_breakdown(pool, round_num)
            all_evolved.extend(pool)

        if not all_evolved:
            logger.warning("Evolution produced no valid instructions, falling back to seeds")
            all_evolved = [
                (s.instruction, {
                    "strategy": "evol_instruct",
                    "seed_index": i,
                    "evolution_round": 0,
                    "evolution_type": "seed_fallback",
                    "evolution_chain": [s.instruction],
                })
                for i, s in enumerate(self._seeds)
            ]

        if self._include_seeds:
            all_evolved = [
                (s.instruction, {
                    "strategy": "evol_instruct",
                    "seed_index": i,
                    "evolution_round": 0,
                    "evolution_type": "seed",
                    "evolution_chain": [s.instruction],
                })
                for i, s in enumerate(self._seeds)
            ] + all_evolved

        self._evolved_items = all_evolved
        logger.info("Evolution complete: %d evolved instructions", len(all_evolved))
        return all_evolved

    def _run_evolution_round(
        self, pool: list[tuple[str, dict]], round_num: int,
    ) -> list[tuple[str, dict]]:
        if self._evol_concurrency <= 1:
            return self._evolve_pool_sequential(pool, round_num)
        return self._evolve_pool_parallel(pool, round_num)

    def _evolve_pool_sequential(self, pool, round_num):
        next_pool = []
        for inst, meta in pool:
            items = self._evolve_one(inst, meta, round_num)
            next_pool.extend(items)
        return next_pool

    def _evolve_pool_parallel(self, pool, round_num):
        next_pool: list[tuple[str, dict]] = []
        with ThreadPoolExecutor(max_workers=self._evol_concurrency) as executor:
            futures = {}
            for inst, meta in pool:
                f = executor.submit(self._evolve_one, inst, meta, round_num)
                futures[f] = (inst, meta)
            for f in as_completed(futures):
                try:
                    items = f.result()
                    next_pool.extend(items)
                except Exception as e:
                    inst, meta = futures[f]
                    logger.warning("Evolution failed for '%s...': %s", inst[:60], e)
        return next_pool

    def _evolve_one(
        self, instruction: str, meta: dict, round_num: int,
    ) -> list[tuple[str, dict]]:
        results: list[tuple[str, dict]] = []

        # Depth evolution
        if random.random() < self._depth_rate and self._depth_mutations:
            mutation = random.choice(self._depth_mutations)
            evolved = self._evolve_depth(instruction, mutation)
            if self._is_valid(instruction, evolved):
                new_meta = self._child_meta(meta, round_num, "depth", mutation["name"])
                new_meta["evolution_chain"] = list(meta["evolution_chain"]) + [evolved]
                results.append((evolved, new_meta))

        # Breadth evolution
        for _ in range(self._branch_factor):
            evolved = self._evolve_breadth(instruction)
            if self._is_valid(instruction, evolved):
                new_meta = self._child_meta(meta, round_num, "breadth", None)
                new_meta["evolution_chain"] = list(meta["evolution_chain"]) + [evolved]
                results.append((evolved, new_meta))

        return results

    @staticmethod
    def _child_meta(parent_meta: dict, round_num: int, etype: str, mutation: Optional[str]) -> dict:
        m = {
            "strategy": "evol_instruct",
            "seed_index": parent_meta.get("seed_index"),
            "evolution_round": round_num,
            "evolution_type": etype,
            "evolution_chain": [],
        }
        if mutation:
            m["mutation"] = mutation
        return m

    def _evolve_depth(self, instruction: str, mutation: dict) -> Optional[str]:
        try:
            prompt = PromptBuilder(lang=self._lang)
            prompt.from_template("evol_system.j2", evolution_type="depth")
            prompt.from_template(
                "evol_depth_user.j2",
                instruction=instruction,
                mutation=mutation["prompt"],
            )
            messages = prompt.build()
            raw = self._call_api(messages, use_json_mode=False,
                                 temperature=self._evol_temperature,
                                 max_tokens=self._evol_max_tokens)
            return self._clean_evolved(raw)
        except Exception as e:
            logger.warning("Depth evolution call failed: %s", e)
            return None

    def _evolve_breadth(self, instruction: str) -> Optional[str]:
        try:
            prompt = PromptBuilder(lang=self._lang)
            prompt.from_template("evol_system.j2", evolution_type="breadth")
            prompt.from_template("evol_breadth_user.j2", instruction=instruction)
            messages = prompt.build()
            raw = self._call_api(messages, use_json_mode=False,
                                 temperature=self._evol_temperature,
                                 max_tokens=self._evol_max_tokens)
            return self._clean_evolved(raw)
        except Exception as e:
            logger.warning("Breadth evolution call failed: %s", e)
            return None

    @staticmethod
    def _clean_evolved(text: str) -> str:
        text = text.strip()
        for q in ('"', "'", "\u201c", "\u201d", "\u300c", "\u300d"):
            if text.startswith(q) and text.endswith(q):
                text = text[1:-1].strip()
        return text

    def _is_valid(self, original: str, evolved: Optional[str]) -> bool:
        if not evolved or evolved == original:
            return False
        ratio = len(evolved) / max(len(original), 1)
        if ratio < self._min_ratio or ratio > self._max_ratio:
            return False
        lower = evolved.lower()
        for phrase in self._REFUSAL_PHRASES:
            if phrase in lower:
                return False
        return True

    def _log_pool_breakdown(self, pool, round_num):
        depths = sum(1 for _, m in pool if m.get("evolution_type") == "depth")
        breadths = sum(1 for _, m in pool if m.get("evolution_type") == "breadth")
        logger.info(
            "Round %d pool: %d items (depth=%d breadth=%d)",
            round_num, len(pool), depths, breadths,
        )

    # ── Phase 2: answer generation ──────────────────────────────────

    def _generate_outputs(self, evolved_items) -> Iterator[GenerationSample]:
        for i, (inst, meta) in enumerate(evolved_items):
            builder = PromptBuilder(lang=self._lang)
            builder.from_template("evol_answer_system.j2")
            builder.from_template(
                "evol_answer_user.j2",
                instruction=inst,
                require_reasoning=self._require_reasoning,
            )
            messages = builder.build()

            try:
                samples = retry_with_backoff(
                    lambda: self._call_and_parse(
                        messages, dict(meta),
                        temperature=self._answer_temperature,
                        max_tokens=self._answer_max_tokens,
                    ),
                    RetryConfig(max_retries=3),
                    f"Evol answer {i}",
                )
            except RuntimeError as e:
                logger.warning("Answer generation failed for evolution item %d: %s", i, e)
                continue

            if samples is None:
                continue
            for s in samples:
                if (s.instruction and s.output) or s.is_multi_turn:
                    yield s

    def estimated_count(self) -> int:
        base = len(self._seeds)
        total = base
        current = base
        for _ in range(self._max_rounds):
            new = int(current * (self._depth_rate + self._branch_factor))
            total += new
            current = new
        return total
