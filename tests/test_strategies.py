import random

from alembic.prompts.builder import load_seeds
from alembic.strategies.seed_driven import SeedDrivenStrategy
from alembic.strategies.topic_driven import TopicDrivenStrategy


class TestTopicDriven:
    def test_estimated_count(self, fake_api):
        strategy = TopicDrivenStrategy(fake_api, {"topics": ["Python", "ML"], "samples_per_topic": 2})
        assert strategy.estimated_count() == 4

    def test_multi_turn_generates_messages(self, fake_multi_turn_api):
        strategy = TopicDrivenStrategy(fake_multi_turn_api, {
            "topics": ["Python"], "samples_per_topic": 1, "multi_turn": True,
        })
        samples = list(strategy.generate())
        assert len(samples) == 1
        assert samples[0].is_multi_turn
        assert len(samples[0].messages) == 4
        assert samples[0].messages[0]["role"] == "user"

    def test_multi_turn_template_selection(self, fake_multi_turn_api):
        strategy = TopicDrivenStrategy(fake_multi_turn_api, {
            "topics": ["Python"], "samples_per_topic": 1, "multi_turn": True,
        })
        prompts = list(strategy.iter_prompts())
        assert len(prompts) == 1
        _pid, messages = prompts[0]
        assert len(messages) >= 1
        assert "multi-turn" in messages[0]["content"].lower()

    def test_batch_one_prompt_per_topic(self, fake_batch_api):
        strategy = TopicDrivenStrategy(fake_batch_api, {
            "topics": ["Math", "Physics"],
            "samples_per_topic": 5,
            "execution_max_per_request": 10,
        })
        prompts = list(strategy.iter_prompts())
        assert len(prompts) == 2
        for _pid, messages in prompts:
            user = messages[-1]["content"]
            assert "Generate" in user
            assert "PLAN" in user

    def test_batch_generates_correct_sample_count(self, fake_batch_api):
        strategy = TopicDrivenStrategy(fake_batch_api, {
            "topics": ["Math", "Physics"],
            "samples_per_topic": 5,
        })
        samples = list(strategy.generate())
        assert len(samples) == 10
        for s in samples:
            assert "sub_topic" in s.metadata
            assert "angle" in s.metadata
            assert "difficulty" in s.metadata
            assert "question_type" in s.metadata

    def test_batch_splits_when_exceeds_planning_max(self, fake_batch_api):
        strategy = TopicDrivenStrategy(fake_batch_api, {
            "topics": ["CS"],
            "samples_per_topic": 25,
            "max_samples_per_request": 10,
            "execution_max_per_request": 30,
        })
        prompts = list(strategy.iter_prompts())
        assert len(prompts) == 1
        for _pid, messages in prompts:
            user = messages[-1]["content"]
            assert "Generate" in user
            assert "25" in user
            assert "PLAN" in user

    def test_batch_single_sample_per_topic(self, fake_batch_api):
        strategy = TopicDrivenStrategy(fake_batch_api, {
            "topics": ["Art"],
            "samples_per_topic": 1,
        })
        samples = list(strategy.generate())
        assert len(samples) == 1

    def test_batch_parse_array(self, fake_batch_api):
        import json
        strategy = TopicDrivenStrategy(fake_batch_api, {"topics": ["X"], "samples_per_topic": 1})
        result = strategy._parse(json.dumps([
            {"instruction": "q1", "output": "a1"},
            {"instruction": "q2", "output": "a2"},
        ]))
        assert len(result) == 2
        assert result[0].instruction == "q1"
        assert result[1].instruction == "q2"

    def test_batch_parse_multi_turn_array(self, fake_batch_api):
        import json
        strategy = TopicDrivenStrategy(fake_batch_api, {"topics": ["X"], "samples_per_topic": 1})
        result = strategy._parse(json.dumps([
            {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
            {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
        ]))
        assert len(result) == 2
        assert result[0].is_multi_turn
        assert result[1].is_multi_turn

    def test_batch_multi_turn_generates_correct_count(self, fake_batch_multi_turn_api):
        strategy = TopicDrivenStrategy(fake_batch_multi_turn_api, {
            "topics": ["Python"], "samples_per_topic": 3, "multi_turn": True,
        })
        samples = list(strategy.generate())
        assert len(samples) == 3
        for s in samples:
            assert s.is_multi_turn
            assert len(s.messages) == 4


class TestSeedDriven:
    def test_load_seeds(self, seed_jsonl):
        seeds = load_seeds(seed_jsonl)
        assert len(seeds) == 3

    def test_load_seeds_formats(self, seed_jsonl):
        seeds = load_seeds(seed_jsonl)
        assert seeds[0].instruction == "what is python"
        assert seeds[0].output == "Python is a programming language."
        assert seeds[1].instruction == ""
        assert seeds[1].output == ""
        assert len(seeds[1].messages) == 2
        assert seeds[1].messages[0]["content"] == "explain ML"
        assert seeds[1].messages[1]["content"] == "ML is a subset of AI."
        assert seeds[2].instruction == "how to use git"
        assert seeds[2].output == "Use git clone, git commit, git push."

    def test_load_seeds_multi_turn(self, multi_turn_seed_jsonl):
        seeds = load_seeds(multi_turn_seed_jsonl)
        assert len(seeds) == 1
        assert len(seeds[0].messages) == 4
        assert seeds[0].messages[0]["role"] == "user"
        assert seeds[0].messages[1]["role"] == "assistant"

    def test_estimated_count(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
        })
        assert strategy.estimated_count() == 3

    def test_multi_turn_generates_messages(self, fake_multi_turn_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_multi_turn_api, {
            "seed_file": seed_jsonl, "example_num": 1, "target_count": 1, "multi_turn": True,
        })
        samples = list(strategy.generate())
        assert len(samples) == 1
        assert samples[0].is_multi_turn
        assert len(samples[0].messages) == 4

    def test_no_evolution_defaults_to_fewshot(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
        })
        prompts = list(strategy.iter_prompts())
        assert len(prompts) == 3
        for pid, _ in prompts:
            assert pid.startswith("seed:")

    def test_crossover_mode_produces_crossover_prompts(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 5,
            "evolution": {"crossover_rate": 1.0, "mutate_rate": 0.0},
        })
        prompts = list(strategy.iter_prompts())
        assert len(prompts) == 5
        for pid, messages in prompts:
            assert pid.startswith("seed_crossover:")
            user = messages[-1]["content"]
            assert "Crossover directive" in user
            assert "Sample A" in user
            assert "Sample B" in user

    def test_crossover_compose_mode(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
            "evolution": {"crossover_rate": 1.0, "mutate_rate": 0.0, "crossover_mode": "compose"},
        })
        prompts = list(strategy.iter_prompts())
        for pid, messages in prompts:
            user = messages[-1]["content"]
            assert "Combine the topics" in user

    def test_crossover_falls_back_with_one_seed(self, fake_api, seed_jsonl, tmp_path):
        one_seed = tmp_path / "one.jsonl"
        one_seed.write_text('{"instruction": "q", "output": "a"}\n', encoding="utf-8")
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": str(one_seed), "example_num": 1, "target_count": 3,
            "evolution": {"crossover_rate": 1.0, "mutate_rate": 0.0},
        })
        prompts = list(strategy.iter_prompts())
        for pid, _ in prompts:
            assert pid.startswith("seed:")

    _MUT_DIFFICULTY = [
        {"name": "difficulty", "values": ["beginner", "intermediate", "advanced"],
         "prompt": "Change the difficulty to '{value}'", "override_field": "difficulty"},
    ]
    _MUT_CONSTRAINT = [
        {"name": "constraint", "values": ["be concise", "use examples"],
         "prompt": "Add this constraint: {value}"},
    ]
    _MUT_TONE = [
        {"name": "tone", "values": ["formal", "casual", "academic"],
         "prompt": "Rewrite in a {value} tone"},
    ]

    def test_mutate_mode_produces_mutate_prompts(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 5,
            "evolution": {"crossover_rate": 0.0, "mutate_rate": 1.0,
                          "mutation_types": self._MUT_TONE},
        })
        prompts = list(strategy.iter_prompts())
        assert len(prompts) == 5
        for pid, messages in prompts:
            assert pid.startswith("seed_mutate:")
            user = messages[-1]["content"]
            assert "Mutation" in user
            assert "Reference sample" in user

    def test_mutate_difficulty_override(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 20,
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": self._MUT_DIFFICULTY,
            },
        })
        prompts = list(strategy.iter_prompts())
        for pid, messages in prompts:
            assert pid.startswith("seed_mutate:")
            user = messages[-1]["content"]
            assert "Change the difficulty to" in user

    def test_mutate_constraint_type(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 5,
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": self._MUT_CONSTRAINT,
            },
        })
        prompts = list(strategy.iter_prompts())
        for pid, messages in prompts:
            user = messages[-1]["content"]
            assert "constraint" in user.lower()

    def test_mixed_rates_produce_all_modes(self, fake_api, seed_jsonl):
        random.seed(42)
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 50,
            "evolution": {"crossover_rate": 0.3, "mutate_rate": 0.3,
                          "mutation_types": self._MUT_TONE},
        })
        prompts = list(strategy.iter_prompts())
        modes = {pid.split(":")[0] for pid, _ in prompts}
        assert "seed" in modes
        assert "seed_crossover" in modes
        assert "seed_mutate" in modes

    def test_evolution_metadata(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 5,
            "evolution": {"crossover_rate": 1.0, "mutate_rate": 0.0, "crossover_mode": "compose"},
        })
        meta = strategy._build_metadata("seed_crossover:0")
        assert meta["evolution"] == "crossover"
        assert meta["crossover_mode"] == "compose"
        meta2 = strategy._build_metadata("seed_mutate:0")
        assert meta2["evolution"] == "mutate"
        meta3 = strategy._build_metadata("seed:0")
        assert "evolution" not in meta3

    def test_rates_normalized_when_exceeding_one(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
            "evolution": {"crossover_rate": 0.8, "mutate_rate": 0.8},
        })
        assert strategy._crossover_rate + strategy._mutate_rate <= 1.0 + 1e-9
        assert strategy._crossover_rate == 0.5
        assert strategy._mutate_rate == 0.5

    def test_crossover_multi_turn(self, fake_multi_turn_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_multi_turn_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
            "multi_turn": True,
            "evolution": {"crossover_rate": 1.0, "mutate_rate": 0.0},
        })
        prompts = list(strategy.iter_prompts())
        for pid, messages in prompts:
            assert pid.startswith("seed_crossover:")
            user = messages[-1]["content"]
            assert "Conversation A" in user
            assert "Conversation B" in user
            assert "multi-turn" in user.lower()

    def test_mutate_multi_turn(self, fake_multi_turn_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_multi_turn_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
            "multi_turn": True,
            "evolution": {"crossover_rate": 0.0, "mutate_rate": 1.0,
                          "mutation_types": self._MUT_TONE},
        })
        prompts = list(strategy.iter_prompts())
        for pid, messages in prompts:
            assert pid.startswith("seed_mutate:")
            user = messages[-1]["content"]
            assert "Reference conversation" in user

    def test_mutate_without_mutation_types_falls_back(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 5,
            "evolution": {"crossover_rate": 0.0, "mutate_rate": 1.0},
        })
        prompts = list(strategy.iter_prompts())
        for pid, _ in prompts:
            assert pid.startswith("seed:")

    def test_custom_mutation_with_values(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 5,
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": [
                    {"name": "tone", "values": ["formal", "casual", "academic"],
                     "prompt": "Rewrite in a {value} tone"},
                ],
            },
        })
        prompts = list(strategy.iter_prompts())
        assert len(prompts) == 5
        for pid, messages in prompts:
            assert pid.startswith("seed_mutate:")
            assert "tone" in pid
            user = messages[-1]["content"]
            assert "Rewrite in a" in user
            assert "tone" in user

    def test_custom_mutation_static_prompt(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": [
                    {"name": "simplify", "prompt": "Simplify the instruction to be more basic"},
                ],
            },
        })
        prompts = list(strategy.iter_prompts())
        for pid, messages in prompts:
            assert "simplify" in pid
            user = messages[-1]["content"]
            assert "Simplify the instruction" in user

    def test_custom_mutation_zh_prompt(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
            "lang": "zh",
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": [
                    {"name": "domain", "values": ["金融", "医疗", "法律"],
                     "prompt": "将问题改写为{value}领域的版本"},
                ],
            },
        })
        prompts = list(strategy.iter_prompts())
        for pid, messages in prompts:
            assert "domain" in pid
            user = messages[-1]["content"]
            assert "领域" in user

    def test_multiple_custom_mutations(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 30,
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": [
                    {"name": "difficulty", "values": ["easy", "hard"],
                     "prompt": "Change difficulty to {value}"},
                    {"name": "tone", "values": ["formal", "casual"],
                     "prompt": "Rewrite in a {value} tone"},
                ],
            },
        })
        prompts = list(strategy.iter_prompts())
        names = {pid.split(":")[2] for pid, _ in prompts}
        assert "difficulty" in names
        assert "tone" in names

    def test_custom_mutation_override_difficulty(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 5,
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": [
                    {"name": "custom_diff", "values": ["easy", "hard"],
                     "prompt": "Set difficulty to {value}",
                     "override_field": "difficulty"},
                ],
            },
        })
        prompts = list(strategy.iter_prompts())
        for pid, messages in prompts:
            user = messages[-1]["content"]
            assert "Set difficulty to" in user
            assert "easy" in user or "hard" in user

    def test_string_entry_skipped(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": ["difficulty"],
            },
        })
        prompts = list(strategy.iter_prompts())
        for pid, _ in prompts:
            assert pid.startswith("seed:")

    def test_custom_mutation_no_prompt_skipped(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": [
                    {"name": "bad", "values": ["x"]},
                    {"name": "good", "prompt": "Always use this mutation"},
                ],
            },
        })
        prompts = list(strategy.iter_prompts())
        for pid, messages in prompts:
            assert "good" in pid
            assert "bad" not in pid

    def test_mutation_type_in_metadata(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
            "evolution": {
                "crossover_rate": 0.0,
                "mutate_rate": 1.0,
                "mutation_types": self._MUT_DIFFICULTY,
            },
        })
        meta = strategy._build_metadata("seed_mutate:0:difficulty")
        assert meta["evolution"] == "mutate"
        assert meta["mutation_type"] == "difficulty"
        meta2 = strategy._build_metadata("seed_mutate:0")
        assert meta2["evolution"] == "mutate"
        assert "mutation_type" not in meta2
