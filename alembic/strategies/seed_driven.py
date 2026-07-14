import logging
import random
from typing import Iterator, Optional

from alembic.api.base import BaseAPIClient
from alembic.core.types import SeedSample, random_topic
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
        self._fixed_topic = params.get("topic")
        self._topic = self._fixed_topic or random_topic()
        self._example_num = max(1, min(int(params.get("example_num", 3)), len(self._seeds)))
        self._target_count = int(params.get("target_count", 10))
        self._multi_turn = bool(params.get("multi_turn", False))

        evo = params.get("evolution") or {}
        self._crossover_rate = self._clamp(float(evo.get("crossover_rate", 0.0)))
        self._mutate_rate = self._clamp(float(evo.get("mutate_rate", 0.0)))
        total = self._crossover_rate + self._mutate_rate
        if total > 1.0:
            self._crossover_rate /= total
            self._mutate_rate /= total
        self._crossover_mode = evo.get("crossover_mode", "instruction_output")
        self._mutation_defs: list[dict] = self._resolve_mutations(evo.get("mutation_types"))
        if self._mutate_rate > 0 and not self._mutation_defs:
            logger.warning(
                "mutate_rate=%.2f but no valid mutation_types configured; "
                "mutation will fall back to default few-shot",
                self._mutate_rate,
            )

    def _get_topic(self) -> str:
        return self._fixed_topic or random_topic()

    def _resolve_mutations(self, raw_types: Optional[list]) -> list[dict]:
        if not raw_types:
            return []
        defs: list[dict] = []
        for entry in raw_types:
            if not isinstance(entry, dict):
                logger.warning("mutation_types entry must be a dict, skipping: %s", entry)
                continue
            name = entry.get("name", "custom")
            prompt = entry.get("prompt", "")
            if not prompt:
                logger.warning("Mutation '%s' has no prompt, skipping", name)
                continue
            d: dict = {"name": name, "prompt": prompt}
            values = entry.get("values")
            if values:
                d["values"] = list(values)
            of = entry.get("override_field")
            if of in ("difficulty", "question_type"):
                d["override_field"] = of
            defs.append(d)
        return defs

    @staticmethod
    def _clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    def _seed_io(self, seed: SeedSample) -> tuple[str, str]:
        if seed.instruction and seed.output:
            return seed.instruction, seed.output
        if seed.messages:
            users = [m["content"] for m in seed.messages if m.get("role") == "user"]
            assistants = [m["content"] for m in seed.messages if m.get("role") == "assistant"]
            if users and assistants:
                return users[0], assistants[0]
        return "", ""

    def _format_seed(self, seed: SeedSample, label: str) -> str:
        if self._multi_turn:
            if seed.messages:
                turns = "\n    ".join(f"[{m['role']}]: {m['content']}" for m in seed.messages)
                return f"{label}:\n    {turns}"
            instr, out = self._seed_io(seed)
            return f"{label}:\n    [user]: {instr}\n    [assistant]: {out}"
        instr, out = self._seed_io(seed)
        return f"{label}:\n  instruction: {instr}\n  output: {out}"

    def _pick_mode(self) -> str:
        r = random.random()
        if r < self._crossover_rate:
            return "crossover"
        if r < self._crossover_rate + self._mutate_rate:
            return "mutate"
        return "default"

    def _build_crossover(self) -> Optional[tuple[str, str]]:
        if len(self._seeds) < 2:
            logger.warning("Crossover requires >=2 seeds, falling back to default")
            return None
        a, b = random.sample(self._seeds, 2)
        noun = "Conversation" if self._multi_turn else "Sample"
        ex_a = self._format_seed(a, f"{noun} A")
        ex_b = self._format_seed(b, f"{noun} B")
        return f"{ex_a}\n\n{ex_b}", self._crossover_directive(noun)

    def _crossover_directive(self, noun: str) -> str:
        if self._crossover_mode == "compose":
            return (
                f"Combine the topics of {noun} A and {noun} B into a single composite "
                f"{'multi-turn conversation that weaves together themes from both' if self._multi_turn else 'instruction; the output should address both topics'}."
            )
        if self._multi_turn:
            return (
                f"Open the conversation with a question similar to {noun} A; let the "
                f"assistant's response style and follow-up pattern follow {noun} B."
            )
        return (
            f"Use {noun} A as the instruction source (the user request) and {noun} B as "
            f"the output style reference; generate a new sample whose instruction solves "
            f"a problem similar to A and whose output is written in the style of B."
        )

    @staticmethod
    def _weighted_choice(defs: list[dict]) -> dict:
        weights = [d.get("weight", 1) for d in defs]
        total = sum(weights)
        if total <= 0:
            return random.choice(defs)
        r = random.random() * total
        for d, w in zip(defs, weights):
            r -= w
            if r <= 0:
                return d
        return defs[-1]

    def _build_mutate(self) -> Optional[tuple[str, dict, Optional[str]]]:
        if not self._seeds or not self._mutation_defs:
            return None
        mdef = self._weighted_choice(self._mutation_defs)
        name = mdef["name"]
        prompt_tmpl = mdef["prompt"]
        values = mdef.get("values")
        template_vars: dict = {}
        chosen_value: Optional[str] = None
        if values:
            value = random.choice(values)
            mutation_str = prompt_tmpl.replace("{value}", str(value))
            chosen_value = str(value)
            of = mdef.get("override_field")
            if of == "difficulty":
                template_vars["override_difficulty"] = value
            elif of == "question_type":
                template_vars["override_question_type"] = value
        else:
            mutation_str = prompt_tmpl
        template_vars["mutation"] = mutation_str
        seed = random.choice(self._seeds)
        label = "Reference conversation" if self._multi_turn else "Reference sample"
        template_vars["examples"] = self._format_seed(seed, label)
        return name, template_vars, chosen_value

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        if not self._seeds or self._example_num == 0:
            logger.warning("No seeds loaded, skipping SeedDrivenStrategy")
            return
        suffix = "_mt" if self._multi_turn else ""
        for i in range(self._target_count):
            mode = self._pick_mode()
            if mode == "crossover":
                built = self._build_crossover()
                if built is None:
                    mode = "default"
                else:
                    examples, directive = built
                    builder = PromptBuilder(lang=self._lang)
                    t = self._get_topic()
                    builder.from_template(f"seed_system{suffix}.j2", topic=t)
                    builder.from_template(
                        f"seed_crossover_user{suffix}.j2",
                        examples=examples,
                        crossover_directive=directive,
                    )
                    yield (f"seed_crossover:{i}", builder.build())
                    continue
            if mode == "mutate":
                built = self._build_mutate()
                if built is None:
                    mode = "default"
                else:
                    mname, template_vars, chosen_value = built
                    pid = f"seed_mutate:{i}:{mname}"
                    if chosen_value:
                        pid += f":{chosen_value}"
                    builder = PromptBuilder(lang=self._lang)
                    t = self._get_topic()
                    builder.from_template(f"seed_system{suffix}.j2", topic=t)
                    builder.from_template(
                        f"seed_mutate_user{suffix}.j2",
                        **template_vars,
                    )
                    yield (pid, builder.build())
                    continue
            chosen = random.sample(self._seeds, min(self._example_num, len(self._seeds)))
            examples_text_parts = []
            for j, seed in enumerate(chosen, 1):
                if seed.messages:
                    turns = "\n    ".join(f"[{m['role']}]: {m['content']}" for m in seed.messages)
                    examples_text_parts.append(f"Example {j}:\n    {turns}")
                else:
                    examples_text_parts.append(f"Example {j}:\n  instruction: {seed.instruction}\n  output: {seed.output}")
            examples_text = "\n\n".join(examples_text_parts)
            t = self._get_topic()
            builder = PromptBuilder(lang=self._lang)
            builder.from_template(f"seed_system{suffix}.j2", topic=t)
            builder.from_template(f"seed_user{suffix}.j2", examples=examples_text)
            yield (f"seed:{i}", builder.build())

    def _build_metadata(self, prompt_id: str) -> dict:
        meta = {"strategy": "seed_driven", "topic": self._get_topic()}
        if prompt_id.startswith("seed_crossover:"):
            meta["evolution"] = "crossover"
            meta["crossover_mode"] = self._crossover_mode
        elif prompt_id.startswith("seed_mutate:"):
            meta["evolution"] = "mutate"
            parts = prompt_id.split(":")
            # parts: ["seed_mutate", idx, type_name, ?value]
            if len(parts) >= 3:
                meta["mutation_type"] = parts[2]
            if len(parts) >= 4:
                meta["mutation_value"] = parts[3]
        return meta

    def estimated_count(self) -> int:
        return self._target_count
