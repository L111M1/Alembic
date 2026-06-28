from alembic.config import AppConfig


class TestAppConfig:
    def test_from_yaml_loads_all_sections(self, sample_config_yaml):
        cfg = AppConfig.from_yaml(sample_config_yaml)
        assert cfg.api.model == "qwen-plus"
        assert len(cfg.strategies) == 3
        assert cfg.cleaner.remove_html is True
        assert cfg.cleaner.minhash_dedup is True
        assert cfg.quality.output_min_len == 10
        assert cfg.scoring.enabled is False
        assert len(cfg.scoring.dimensions) == 4
        assert cfg.output.multi_turn is False

    def test_api_config_defaults(self):
        cfg = AppConfig()
        assert cfg.api.model == "gpt-4o"
        assert cfg.api.concurrency == 1

    def test_scoring_config_section(self, sample_config_yaml):
        cfg = AppConfig.from_yaml(sample_config_yaml)
        assert cfg.scoring.enabled is False
        assert len(cfg.scoring.dimensions) == 4
