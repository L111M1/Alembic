from alembic.prompts.builder import load_seeds
from alembic.strategies.base import GenerationStrategy
from alembic.strategies.seed_driven import SeedDrivenStrategy
from alembic.strategies.self_instruct import SelfInstructStrategy
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


class TestSelfInstruct:
    def test_estimated_count(self, fake_api):
        strategy = SelfInstructStrategy(fake_api, {"target_count": 5})
        assert strategy.estimated_count() == 5

    def test_multi_turn_generates_messages(self, fake_multi_turn_api):
        strategy = SelfInstructStrategy(fake_multi_turn_api, {
            "target_count": 1, "multi_turn": True,
        })
        samples = list(strategy.generate())
        assert len(samples) == 1
        assert samples[0].is_multi_turn
        assert len(samples[0].messages) == 4
