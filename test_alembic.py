"""Integration test for alembic framework"""
import sys
sys.path.insert(0, '.')

from alembic.strategies.topic_driven import TopicDrivenStrategy
from alembic.strategies.seed_driven import SeedDrivenStrategy
from alembic.strategies.self_instruct import SelfInstructStrategy
from alembic.strategies.composite import create_strategy, CompositeStrategy
from alembic.prompts.builder import PromptBuilder, load_seeds
from alembic.quality.validators import build_validator_chain
from alembic.config import QualityConfig, CleanerConfig, AppConfig
from alembic.api.base import BaseAPIClient
from alembic.core.types import GenerationSample
from alembic.cleaner import DatasetCleaner
import tempfile
import os


class FakeAPI(BaseAPIClient):
    def supports_json_mode(self):
        return True

    def call(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        return '{"instruction": "test instruction here", "output": "test output data here for testing"}'


def main():
    api = FakeAPI()

    topics = TopicDrivenStrategy(api, {'topics': ['Python', 'ML'], 'samples_per_topic': 2})
    assert topics.estimated_count() == 4
    print("Test 1 (TopicDriven): PASSED")

    seeds = load_seeds('seeds.jsonl')
    assert len(seeds) == 3
    print(f"Test 2 (Seeds): PASSED ({len(seeds)})")

    seed_strat = SeedDrivenStrategy(api, {'seed_file': 'seeds.jsonl', 'example_num': 2, 'target_count': 3})
    assert seed_strat.estimated_count() == 3
    print("Test 3 (SeedDriven): PASSED")

    si = SelfInstructStrategy(api, {'target_count': 5})
    assert si.estimated_count() == 5
    print("Test 4 (SelfInstruct): PASSED")

    chain = build_validator_chain(QualityConfig())
    s = GenerationSample(instruction='test instruction here', output='test output data here')
    assert chain.validate(s)
    bad = GenerationSample(instruction='', output='')
    assert not chain.validate(bad)
    print("Test 5 (Quality): PASSED")

    composite = create_strategy(api, [
        {'type': 'topic_driven', 'weight': 0.6, 'topics': ['A'], 'samples_per_topic': 3},
        {'type': 'self_instruct', 'weight': 0.4, 'target_count': 2},
    ])
    assert isinstance(composite, CompositeStrategy)
    gen = composite.generate()
    s1 = next(gen)
    assert s1.instruction == 'test instruction here'
    print("Test 6 (Composite+Generate): PASSED")

    cfg = AppConfig.from_yaml('sft_gen_config.yaml')
    assert cfg.api.model == 'qwen-plus'
    assert len(cfg.strategies) == 3
    assert cfg.cleaner.remove_html == True
    print(f"Test 7 (Config): PASSED")

    builder = PromptBuilder()
    builder.system("You are helpful")
    builder.user("Hello")
    msgs = builder.build()
    assert len(msgs) == 2
    print("Test 8 (PromptBuilder): PASSED")

    builder2 = PromptBuilder()
    builder2.from_template("topic_driven_system.j2")
    builder2.from_template("topic_driven_user.j2", topic="Python")
    msgs2 = builder2.build()
    assert len(msgs2) == 2
    assert 'Python' in msgs2[1]['content']
    print("Test 9 (Template): PASSED")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        f.write('{"instruction": "what is python", "output": "Python is a high-level programming language used for web development and data science."}\n')
        f.write('{"instruction": "<html>test html</html>", "output": "visit http://example.com for info"}\n')
        f.write('{"instruction": "", "output": ""}\n')
        tmp_in = f.name
    tmp_out = tmp_in.replace('.jsonl', '_cleaned.jsonl')

    cleaner_cfg = CleanerConfig(remove_html=True, remove_urls=True, output_min_len=10, output_max_len=8000, dedup=False)
    cleaner = DatasetCleaner(cleaner_cfg)
    kept, dropped = cleaner.clean_file(tmp_in, tmp_out)
    assert kept == 2
    assert dropped == 1
    os.unlink(tmp_in)
    os.unlink(tmp_out)
    print("Test 10 (Cleaner): PASSED (kept=1, dropped=2)")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == '__main__':
    main()
