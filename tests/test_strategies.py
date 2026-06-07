from alembic.strategies.topic_driven import TopicDrivenStrategy
from alembic.strategies.seed_driven import SeedDrivenStrategy
from alembic.strategies.self_instruct import SelfInstructStrategy
from alembic.prompts.builder import load_seeds


class TestTopicDriven:
    def test_estimated_count(self, fake_api):
        strategy = TopicDrivenStrategy(fake_api, {"topics": ["Python", "ML"], "samples_per_topic": 2})
        assert strategy.estimated_count() == 4


class TestSeedDriven:
    def test_load_seeds(self, seed_jsonl):
        seeds = load_seeds(seed_jsonl)
        assert len(seeds) == 3

    def test_load_seeds_formats(self, seed_jsonl):
        seeds = load_seeds(seed_jsonl)
        assert seeds[0].instruction == "what is python"
        assert seeds[0].output == "Python is a programming language."
        assert seeds[1].instruction == "explain ML"
        assert seeds[1].output == "ML is a subset of AI."
        assert seeds[2].instruction == "how to use git"
        assert seeds[2].output == "Use git clone, git commit, git push."

    def test_estimated_count(self, fake_api, seed_jsonl):
        strategy = SeedDrivenStrategy(fake_api, {
            "seed_file": seed_jsonl, "example_num": 2, "target_count": 3,
        })
        assert strategy.estimated_count() == 3


class TestSelfInstruct:
    def test_estimated_count(self, fake_api):
        strategy = SelfInstructStrategy(fake_api, {"target_count": 5})
        assert strategy.estimated_count() == 5
