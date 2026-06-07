from alembic.quality.validators import build_validator_chain
from alembic.config import QualityConfig
from alembic.core.types import GenerationSample


class TestQuality:
    def test_chain_validates_good_sample(self):
        chain = build_validator_chain(QualityConfig())
        sample = GenerationSample(
            instruction="test instruction here",
            output="test output data here",
        )
        assert chain.validate(sample)

    def test_chain_rejects_empty_sample(self):
        chain = build_validator_chain(QualityConfig())
        bad = GenerationSample(instruction="", output="")
        assert not chain.validate(bad)

    def test_chain_rejects_short_instruction(self):
        cfg = QualityConfig(instruction_min_len=20)
        chain = build_validator_chain(cfg)
        sample = GenerationSample(instruction="short", output="long enough output here for testing")
        assert not chain.validate(sample)

    def test_chain_rejects_short_output(self):
        cfg = QualityConfig(output_min_len=100)
        chain = build_validator_chain(cfg)
        sample = GenerationSample(instruction="valid instruction", output="short")
        assert not chain.validate(sample)
