import json
import random

import pytest

from alembic.api.base import BaseAPIClient
from alembic.strategies.evol_instruct import EvolInstructStrategy


class FakeEvolAPI(BaseAPIClient):
    """Fake API that simulates evolution + answer generation."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        self.call_count += 1
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        sys_lower = system.lower()

        if "evolution engine" in sys_lower or "指令进化引擎" in sys_lower:
            return f"Evolved instruction v{self.call_count}"
        if "prompt creator" in sys_lower or "提示词创作者" in sys_lower:
            return f"Breadth instruction v{self.call_count}"

        return json.dumps({
            "instruction": "test evolved instruction",
            "output": "test answer output",
        })


class FakeFailingEvolAPI(BaseAPIClient):
    """Fake API where evolution returns empty string (invalid)."""

    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        sys_lower = system.lower()

        if "evolution engine" in sys_lower or "指令进化引擎" in sys_lower:
            return ""  # empty = invalid
        if "prompt creator" in sys_lower or "提示词创作者" in sys_lower:
            return ""  # empty = invalid

        return json.dumps({
            "instruction": "fallback instruction",
            "output": "fallback answer",
        })


@pytest.fixture
def fake_evol_api():
    return FakeEvolAPI()


@pytest.fixture
def fake_failing_api():
    return FakeFailingEvolAPI()


@pytest.fixture
def seed_path(temp_jsonl):
    return temp_jsonl([
        json.dumps({"instruction": "What is Python?", "output": "Python is a programming language."}),
        json.dumps({"instruction": "Explain recursion", "output": "Recursion is a technique where..."}),
    ])


class TestEvolInstruct:
    def test_requires_seed_file(self, fake_evol_api):
        with pytest.raises(ValueError, match="seed_file"):
            EvolInstructStrategy(fake_evol_api, {})

    def test_requires_valid_seeds(self, fake_evol_api, temp_jsonl):
        path = temp_jsonl([
            json.dumps({"instruction": "", "output": "empty instruction"}),
        ])
        with pytest.raises(ValueError, match="No seeds with valid instructions"):
            EvolInstructStrategy(fake_evol_api, {"seed_file": path})

    def test_estimated_count(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 2,
            "depth_rate": 0.5,
            "branch_factor": 0,
        })
        assert strategy.estimated_count() > 0

    def test_basic_evolution_generates_samples(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 2,
            "depth_rate": 1.0,
            "branch_factor": 0,
        })
        samples = list(strategy.generate())
        assert len(samples) > 0
        for s in samples:
            assert s.instruction
            assert s.output
            assert s.metadata is not None
            assert s.metadata.get("strategy") == "evol_instruct"

    def test_evolution_metadata_chain(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 1.0,
            "branch_factor": 0,
        })
        samples = list(strategy.generate())
        assert len(samples) > 0
        for s in samples:
            m = s.metadata
            assert "evolution_chain" in m
            assert len(m["evolution_chain"]) >= 1
            assert "evolution_round" in m
            assert "evolution_type" in m
            assert m["evolution_type"] in ("seed", "depth", "breadth", "seed_fallback")

    def test_breadth_evolution_adds_variants(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 0.0,
            "branch_factor": 2,
        })
        samples = list(strategy.generate())
        assert len(samples) >= 2

    def test_no_output_generation(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 1.0,
            "branch_factor": 0,
            "generate_output": False,
        })
        samples = list(strategy.generate())
        assert len(samples) > 0
        for s in samples:
            assert s.instruction
            assert not s.output

    def test_fallback_to_seeds_on_evolution_failure(self, fake_failing_api, seed_path):
        strategy = EvolInstructStrategy(fake_failing_api, {
            "seed_file": seed_path,
            "max_rounds": 2,
            "depth_rate": 1.0,
            "branch_factor": 0,
        })
        samples = list(strategy.generate())
        assert len(samples) >= 1

    def test_include_seeds(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 1.0,
            "branch_factor": 0,
            "include_seeds": True,
        })
        samples = list(strategy.generate())
        seed_instructions = ["What is Python?", "Explain recursion"]
        found_seeds = any(
            s.instruction in seed_instructions or m.get("evolution_round") == 0
            for s in samples
            for m in [s.metadata or {}]
        )
        assert found_seeds or len([s for s in samples if s.metadata and s.metadata.get("evolution_round") == 0]) > 0

    def test_chinese_prompts(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 1.0,
            "branch_factor": 0,
            "lang": "zh",
        })
        samples = list(strategy.generate())
        assert len(samples) > 0

    def test_is_valid_rejects_identical(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 1.0,
            "branch_factor": 0,
        })
        assert not strategy._is_valid("hello", "hello")
        assert strategy._is_valid("hello", "hello world")

    def test_is_valid_rejects_empty(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
        })
        assert not strategy._is_valid("hello", "")

    def test_is_valid_rejects_refusal(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
        })
        assert not strategy._is_valid("hello", "sorry, i cannot answer that")

    def test_is_valid_checks_length_ratio(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "min_evolution_ratio": 0.5,
            "max_evolution_ratio": 3.0,
        })
        assert strategy._is_valid("a" * 10, "a" * 20)
        assert not strategy._is_valid("a" * 10, "a" * 2)
        assert not strategy._is_valid("a" * 10, "a" * 100)

    def test_child_meta_preserves_seed_index(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
        })
        parent = {"seed_index": 0}
        child = strategy._child_meta(parent, 2, "depth", "add_constraint")
        assert child["seed_index"] == 0
        assert child["evolution_round"] == 2
        assert child["evolution_type"] == "depth"
        assert child["mutation"] == "add_constraint"

    def test_clean_evolved_removes_quotes(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {"seed_file": seed_path})
        assert strategy._clean_evolved('"hello"') == "hello"
        assert strategy._clean_evolved("'hello'") == "hello"
        assert strategy._clean_evolved("hello") == "hello"

    def test_mixed_depth_and_breadth(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 0.5,
            "branch_factor": 1,
        })
        samples = list(strategy.generate())
        assert len(samples) > 0

    def test_different_evolution_types_in_metadata(self, fake_evol_api, seed_path):
        random.seed(42)
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 0.5,
            "branch_factor": 1,
        })
        strategy._run_evolution()
        types = {meta.get("evolution_type") for _, meta in strategy._evolved_items}
        assert "depth" in types or "breadth" in types

    def test_concurrency_does_not_break(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 1.0,
            "branch_factor": 2,
            "evol_concurrency": 4,
        })
        samples = list(strategy.generate())
        assert len(samples) > 0

    def test_custom_depth_mutations(self, fake_evol_api, seed_path):
        custom_mutations = [
            {"name": "custom_mut", "prompt": "Make it more difficult"},
        ]
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 1.0,
            "branch_factor": 0,
            "depth_mutations": custom_mutations,
        })
        samples = list(strategy.generate())
        assert len(samples) > 0

    def test_require_reasoning(self, fake_evol_api, seed_path):
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 1.0,
            "branch_factor": 0,
            "require_reasoning": True,
        })
        samples = list(strategy.generate())
        assert len(samples) > 0
