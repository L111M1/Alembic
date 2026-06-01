import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from alembic.api.factory import create_client
from alembic.config import AppConfig
from alembic.core.types import GenerationStats
from alembic.core.observer import LogObserver, CompositeObserver
from alembic.quality.validators import build_validator_chain
from alembic.strategies.composite import create_strategy
from alembic.writers.jsonl_writer import JSONLWriter
from alembic.cleaner.cleaner import DatasetCleaner

logger = logging.getLogger(__name__)


class Pipeline:
    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        config = AppConfig.from_yaml(path)
        return cls(config)

    def __init__(self, config: AppConfig):
        self._config = config

    def run(self) -> GenerationStats:
        if self._config.random_seed is not None:
            import random
            random.seed(self._config.random_seed)

        api = create_client(
            model=self._config.api.model,
            api_key=self._config.api.api_key,
            base_url=self._config.api.base_url,
            retry=self._config.api.retry,
        )

        strategy_cfgs = []
        for s in self._config.strategies:
            d = {"type": s.type, "weight": s.weight, "lang": self._config.api.lang, "concurrency": self._config.api.concurrency}
            d.update(s.params)
            strategy_cfgs.append(d)
        strategy = create_strategy(api, strategy_cfgs)

        validator = build_validator_chain(self._config.quality)

        writer = None if self._config.dry_run else JSONLWriter(self._config.output)

        observer = LogObserver(logger)

        stats = GenerationStats()
        max_count = self._config.count if self._config.count > 0 else strategy.estimated_count()

        observer.on_start(max_count)

        try:
            gen = strategy.generate()
            pbar = tqdm(total=max_count, desc="Generating", unit="sample")
            idx = 0
            while idx < max_count:
                try:
                    sample = next(gen)
                except StopIteration:
                    break

                stats.total_attempted += 1

                if validator.validate(sample):
                    stats.total_generated += 1
                    if writer:
                        writer.write(sample)
                    pbar.set_postfix(kept=stats.total_generated, filtered=stats.total_filtered)
                else:
                    stats.total_filtered += 1
                    pbar.set_postfix(kept=stats.total_generated, filtered=stats.total_filtered)

                pbar.update(1)
                idx += 1
            pbar.close()
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        finally:
            if writer:
                writer.close()
            observer.on_complete(stats)

        if not self._config.dry_run and self._config.output.path:
            self._run_cleaner()

        return stats

    def _run_cleaner(self) -> None:
        output_path = self._config.output.path
        if not Path(output_path).exists():
            return

        cleaner = DatasetCleaner(self._config.cleaner)
        cleaned_path = output_path
        if not cleaned_path.endswith("_cleaned.jsonl"):
            cleaned_path = output_path.replace(".jsonl", "_cleaned.jsonl")

        kept, dropped = cleaner.clean_file(output_path, cleaned_path)
        logger.info(f"Cleaner: kept={kept}, dropped={dropped}, result={cleaned_path}")
