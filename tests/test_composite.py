from alembic.registry import create_strategy
from alembic.strategies.composite import CompositeStrategy


class TestComposite:
    def test_create_composite_strategy(self, fake_api, seed_jsonl):
        composite = create_strategy(fake_api, [
            {"type": "topic_driven", "weight": 0.6, "topics": ["A"], "samples_per_topic": 3},
            {"type": "seed_driven", "weight": 0.4, "seed_file": seed_jsonl, "example_num": 2, "target_count": 2},
        ])
        assert isinstance(composite, CompositeStrategy)

    def test_composite_generate_yields_samples(self, fake_api, seed_jsonl):
        composite = create_strategy(fake_api, [
            {"type": "topic_driven", "weight": 0.6, "topics": ["A"], "samples_per_topic": 3},
            {"type": "seed_driven", "weight": 0.4, "seed_file": seed_jsonl, "example_num": 2, "target_count": 2},
        ])
        gen = composite.generate()
        s1 = next(gen)
        assert s1.instruction == "test instruction here"
        assert s1.output == "test output data here for testing"

    def test_create_single_strategy(self, fake_api, seed_jsonl):
        strategy = create_strategy(fake_api, [
            {"type": "seed_driven", "weight": 1.0, "seed_file": seed_jsonl, "example_num": 2, "target_count": 3},
        ])
        assert not isinstance(strategy, CompositeStrategy)
