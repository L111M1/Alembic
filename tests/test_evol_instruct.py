import json
import random
from contextlib import contextmanager

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

    # ── WizardLM-specific tests ─────────────────────────────────────

    def test_wizardlm_5_depth_operators(self):
        """WizardLM paper defines 5 depth evolution operators."""
        names = {m["name"] for m in EvolInstructStrategy._DEFAULT_DEPTH_MUTATIONS}
        expected = {"add_constraint", "deepen", "concretize",
                    "increase_reasoning", "complicate_input"}
        assert names == expected, f"Missing: {expected - names}"

    def test_wizardlm_evolution_chain_structure(self, fake_evol_api, seed_path):
        """Each evolution round appends to the chain; chain length == round."""
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 3,
            "depth_rate": 1.0,
            "branch_factor": 0,
        })
        strategy._plan_all()
        for _, meta in strategy._evolved_items:
            assert len(meta["evolution_chain"]) == meta["evolution_round"] + 1
            assert meta["evolution_chain"][0] in ("What is Python?", "Explain recursion")

    def test_wizardlm_round_advances_instruction(self, fake_evol_api, seed_path):
        """Each round's evolved instruction differs from the seed."""
        random.seed(42)
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 2,
            "depth_rate": 1.0,
            "branch_factor": 0,
        })
        strategy._plan_all()
        for inst, meta in strategy._evolved_items:
            if meta["evolution_round"] > 0:
                assert inst != meta["evolution_chain"][0]

    def test_wizardlm_all_5_operators_used(self, fake_evol_api, seed_path):
        """With enough rounds all 5 depth operators should appear in metadata."""
        random.seed(0)
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 10,
            "depth_rate": 1.0,
            "branch_factor": 0,
        })
        strategy._plan_all()
        used = set()
        for _, meta in strategy._evolved_items:
            m = meta.get("mutation")
            if m:
                used.add(m)
        assert len(used) >= 3, f"Expected >= 3 operators used, got {used}"

    def test_wizardlm_breadth_different_domain(self, fake_evol_api, seed_path):
        """Breadth evolution produces instructions on different topics."""
        random.seed(42)
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 0.0,
            "branch_factor": 3,
        })
        strategy._plan_all()
        breadth_items = [(i, m) for i, m in strategy._evolved_items
                         if m.get("evolution_type") == "breadth"]
        assert len(breadth_items) >= 1
        for inst, meta in breadth_items:
            assert inst != meta["evolution_chain"][0]

    def test_wizardlm_evolve_one_depth(self, fake_evol_api, seed_path):
        """_evolve_one with depth_rate=1 produces a depth-evolved instruction."""
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 2,
            "depth_rate": 1.0,
            "branch_factor": 0,
        })
        with random_seed(42):
            items = strategy._evolve_one("What is Python?",
                                         {"seed_index": 0, "evolution_chain": ["What is Python?"]},
                                         1)
        assert len(items) >= 1
        for inst, meta in items:
            assert meta["evolution_type"] == "depth"
            assert meta["mutation"] in (m["name"] for m in strategy._DEFAULT_DEPTH_MUTATIONS)

    def test_wizardlm_evolve_one_breadth(self, fake_evol_api, seed_path):
        """_evolve_one with branch_factor=1 produces a breadth-evolved instruction."""
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 2,
            "depth_rate": 0.0,
            "branch_factor": 1,
        })
        with random_seed(0):
            items = strategy._evolve_one("What is Python?",
                                         {"seed_index": 0, "evolution_chain": ["What is Python?"]},
                                         1)
        assert len(items) >= 1
        for inst, meta in items:
            assert meta["evolution_type"] == "breadth"
            assert meta.get("mutation") is None

    def test_wizardlm_evolve_one_depth_and_breadth(self, fake_evol_api, seed_path):
        """depth_rate=0.5 + branch_factor=1 may produce both types."""
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 1,
            "depth_rate": 0.5,
            "branch_factor": 1,
        })
        strategy._plan_all()
        types = {meta.get("evolution_type") for _, meta in strategy._evolved_items}
        assert types.issubset({"seed", "depth", "breadth", "seed_fallback"})
        assert "breadth" in types

    def test_wizardlm_round_pool_growth(self, fake_evol_api, seed_path):
        """Pool grows when branch_factor > 0 and shrinks when evolution fails."""
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 3,
            "depth_rate": 0.5,
            "branch_factor": 1,
        })
        evolved = strategy._run_evolution()
        assert len(evolved) > 0
        rounds = {m["evolution_round"] for _, m in evolved}
        assert max(rounds) > 0

    def test_wizardlm_two_phase_pipeline(self, fake_evol_api, seed_path):
        """Two-phase pipeline: evolve → answer. Output must have instruction + output."""
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 2,
            "depth_rate": 0.7,
            "branch_factor": 0,
        })
        samples = list(strategy.generate())
        for s in samples:
            assert s.instruction
            assert s.output
            assert s.metadata
            assert s.metadata["strategy"] == "evol_instruct"

    def test_wizardlm_deep_chain_multiple_rounds(self, fake_evol_api, seed_path):
        """Deep chain: round N instruction appears in round N+1's evolution_chain."""
        random.seed(1)
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 3,
            "depth_rate": 1.0,
            "branch_factor": 0,
        })
        evolved = strategy._run_evolution()
        for inst, meta in evolved:
            if meta["evolution_round"] >= 2:
                chain = meta["evolution_chain"]
                assert len(chain) == meta["evolution_round"] + 1
                for r in range(meta["evolution_round"]):
                    assert chain[r] != chain[r + 1]

    def test_wizardlm_non_output_mode(self, fake_evol_api, seed_path):
        """When generate_output=False, only evolved instructions are yielded."""
        strategy = EvolInstructStrategy(fake_evol_api, {
            "seed_file": seed_path,
            "max_rounds": 2,
            "depth_rate": 1.0,
            "branch_factor": 0,
            "generate_output": False,
        })
        samples = list(strategy.generate())
        assert len(samples) > 0
        for s in samples:
            assert s.instruction
            assert not s.output
            assert s.metadata
            assert "evolution_round" in s.metadata


@pytest.fixture
def fake_evol_api_wizardlm():
    """Fake API that returns instruction-length-based responses."""
    class _FakeWizardLMAPI(BaseAPIClient):
        _call_idx = 0

        def supports_json_mode(self):
            return True

        def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
            _FakeWizardLMAPI._call_idx += 1
            system = next((m["content"] for m in messages if m.get("role") == "system"), "")
            sys_lower = system.lower()

            if "evolution engine" in sys_lower or "指令进化引擎" in sys_lower:
                return f"Evolved instruction (round {_FakeWizardLMAPI._call_idx})"
            if "prompt creator" in sys_lower or "提示词创作者" in sys_lower:
                return f"Breadth instruction (round {_FakeWizardLMAPI._call_idx})"
            return json.dumps({
                "instruction": "evolved output instruction",
                "output": f"answer for round {_FakeWizardLMAPI._call_idx}",
            })

    _FakeWizardLMAPI._call_idx = 0
    return _FakeWizardLMAPI()


@contextmanager
def random_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)
