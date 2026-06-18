import json
import logging
from pathlib import Path

from tqdm import tqdm

from alembic.api.factory import create_client
from alembic.cleaner.cleaner import DatasetCleaner
from alembic.config import AppConfig
from alembic.core.observer import CompositeObserver, LogObserver
from alembic.core.stats import StatisticsCollector
from alembic.core.types import GenerationStats
from alembic.quality.validators import build_validator_chain
from alembic.scoring.scorer import DatasetScorer
from alembic.strategies.composite import create_strategy
from alembic.writers.jsonl_writer import JSONLWriter

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: AppConfig):
        self._config = config
        self._stats_collector = StatisticsCollector()

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        config = AppConfig.from_yaml(path)
        return cls(config)

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
            d = {"type": s.type, "weight": s.weight, "lang": self._config.api.lang, "concurrency": self._config.api.concurrency, "multi_turn": self._config.output.multi_turn}
            d.update(s.params)
            strategy_cfgs.append(d)
        strategy = create_strategy(api, strategy_cfgs)

        validator = build_validator_chain(self._config.quality)

        writer = None if self._config.dry_run else JSONLWriter(self._config.output)

        observer = CompositeObserver(LogObserver(logger), self._stats_collector)

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
                strategy_name = sample.metadata.get("strategy", "unknown") if sample.metadata else "unknown"
                observer.on_sample(idx, True, strategy_name)

                if validator.validate(sample):
                    stats.total_generated += 1
                    stats.by_strategy[strategy_name] = stats.by_strategy.get(strategy_name, 0) + 1
                    if writer:
                        writer.write(sample)
                        record = {"instruction": sample.instruction, "output": sample.output}
                        if sample.is_multi_turn:
                            record["messages"] = sample.messages
                        if sample.metadata:
                            record["metadata"] = sample.metadata
                        self._stats_collector.record_sample(record)
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
            cleaned_path = self._run_cleaner()
            if self._config.scoring.enabled and self._config.scoring.dimensions:
                scored_path = self._run_scorer(cleaned_path)
                if self._config.scoring.min_total_score > 0:
                    self._run_score_filter(scored_path)

            self._stats_collector.save_report(self._config.output.path)

        return stats

    def _run_cleaner(self) -> str:
        output_path = self._config.output.path
        if not Path(output_path).exists():
            return output_path

        cleaner = DatasetCleaner(self._config.cleaner)
        cleaned_path = output_path
        if not cleaned_path.endswith("_cleaned.jsonl"):
            cleaned_path = output_path.replace(".jsonl", "_cleaned.jsonl")

        kept, dropped = cleaner.clean_file(output_path, cleaned_path)
        self._stats_collector.record_cleaner(kept, dropped)
        logger.info(f"Cleaner: kept={kept}, dropped={dropped}, result={cleaned_path}")
        return cleaned_path

    def _run_scorer(self, input_path: str) -> str:
        scfg = self._config.scoring
        output_path = scfg.output_path
        if not output_path:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_scored.jsonl")

        scoring_api = create_client(
            model=scfg.model,
            api_key=scfg.api_key,
            base_url=scfg.base_url,
            retry=scfg.retry,
        )

        scorer = DatasetScorer(scfg)
        scored, failed = scorer.score_file(scoring_api, input_path, output_path)
        self._stats_collector.record_scorer(scored, failed)
        self._stats_collector.record_scores(output_path)
        logger.info(f"Scorer: scored={scored}, failed={failed}, result={output_path}")
        return output_path

    def _run_score_filter(self, input_path: str) -> None:
        min_score = self._config.scoring.min_total_score
        output_path = input_path.replace("_scored.jsonl", "_scored_filtered.jsonl")
        kept = dropped = 0

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                total = sample.get("total_score", 0)
                if total >= min_score:
                    fout.write(json.dumps({"instruction": sample.get("instruction", ""), "output": sample.get("output", ""), "metadata": sample.get("metadata", {}), "scores": sample.get("scores", {}), "total_score": total}, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    dropped += 1

        self._stats_collector.record_score_filter(kept, dropped)
        logger.info(f"Score filter (min={min_score}): kept={kept}, dropped={dropped}, result={output_path}")
