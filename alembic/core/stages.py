import abc
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from alembic.api.base import BaseAPIClient
from alembic.cleaner.cleaner import DatasetCleaner
from alembic.config import AppConfig
from alembic.core.observer import CompositeObserver, LogObserver, Observer
from alembic.core.stats import StatisticsCollector
from alembic.core.types import GenerationStats
from alembic.quality.validators import QualityValidator, build_validator_chain
from alembic.registry import create_client, create_strategy, stage_registry
from alembic.scoring.scorer import DatasetScorer
from alembic.strategies.base import GenerationStrategy
from alembic.writers.jsonl_writer import BaseWriter, JSONLWriter, MemoryWriter

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Mutable carrier passed along the pipeline stages."""
    config: AppConfig
    collector: StatisticsCollector
    stats: GenerationStats = field(default_factory=GenerationStats)
    output_path: str = ""
    samples: list[dict] = field(default_factory=list)


class PipelineStage(abc.ABC):
    """A single stage in the generate -> clean -> score -> filter pipeline."""

    @abc.abstractmethod
    def process(self, ctx: PipelineContext) -> None: ...


def _derive_path(base: str, suffix: str) -> str:
    if base.endswith(f"_{suffix}.jsonl"):
        return base
    return base.replace(".jsonl", f"_{suffix}.jsonl")


def _load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


class GenerationStage(PipelineStage):
    def __init__(
        self,
        api: Optional[BaseAPIClient] = None,
        strategy: Optional[GenerationStrategy] = None,
        validator: Optional[QualityValidator] = None,
        writer: Optional[BaseWriter] = None,
        observer: Optional[Observer] = None,
    ):
        self._api = api
        self._strategy = strategy
        self._validator = validator
        self._writer = writer
        self._observer = observer

    def process(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        api = self._api or self._create_api(cfg)
        strategy = self._strategy or self._create_strategy(api, cfg)
        validator = self._validator or self._create_validator(cfg)
        observer = self._observer or self._create_observer(ctx)

        writer = MemoryWriter(cfg.output.format) if not cfg.dry_run else None
        self._run_generation_loop(cfg, ctx, strategy, validator, writer, observer, api)

        if writer:
            ctx.samples = writer.records
        ctx.output_path = cfg.output.path

    def _create_api(self, cfg: AppConfig) -> BaseAPIClient:
        return create_client(
            model=cfg.api.model,
            api_key=cfg.api.api_key,
            base_url=cfg.api.base_url,
            retry=cfg.api.retry,
        )

    def _create_strategy(self, api: BaseAPIClient, cfg: AppConfig) -> GenerationStrategy:
        strategy_cfgs = []
        for s in cfg.strategies:
            d = {
                "type": s.type,
                "weight": s.weight,
                "lang": cfg.api.lang,
                "concurrency": cfg.api.concurrency,
                "multi_turn": cfg.output.multi_turn,
            }
            d.update(s.params)
            if cfg.count > 0:
                d["total_count"] = min(cfg.count, d.get("total_count", cfg.count))
            strategy_cfgs.append(d)
        return create_strategy(api, strategy_cfgs)

    def _create_validator(self, cfg: AppConfig) -> Optional[QualityValidator]:
        return build_validator_chain(cfg.quality)

    def _create_writer(self, cfg: AppConfig) -> Optional[BaseWriter]:
        return None if cfg.dry_run else JSONLWriter(cfg.output)

    def _create_observer(self, ctx: PipelineContext) -> Observer:
        return CompositeObserver(LogObserver(logger), ctx.collector)

    def _run_generation_loop(
        self,
        cfg: AppConfig,
        ctx: PipelineContext,
        strategy: GenerationStrategy,
        validator: Optional[QualityValidator],
        writer: Optional[BaseWriter],
        observer: Observer,
        api: BaseAPIClient,
    ) -> None:
        stats = ctx.stats
        max_count = cfg.count if cfg.count > 0 else strategy.estimated_count()
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
                strategy_name = (
                    sample.metadata.get("strategy", "unknown")
                    if sample.metadata
                    else "unknown"
                )
                observer.on_sample(idx, True, strategy_name)

                if not validator or validator.validate(sample):
                    stats.total_generated += 1
                    stats.by_strategy[strategy_name] = (
                        stats.by_strategy.get(strategy_name, 0) + 1
                    )
                    if writer:
                        writer.write(sample)
                        record = {
                            "instruction": sample.instruction,
                            "output": sample.output,
                        }
                        if sample.is_multi_turn:
                            record["messages"] = sample.messages
                        if sample.metadata:
                            record["metadata"] = sample.metadata
                        ctx.collector.record_sample(record)
                    pbar.set_postfix(
                        kept=stats.total_generated, filtered=stats.total_filtered
                    )
                else:
                    stats.total_filtered += 1
                    pbar.set_postfix(
                        kept=stats.total_generated, filtered=stats.total_filtered
                    )

                pbar.update(1)
                idx += 1
            pbar.close()
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        finally:
            if writer:
                writer.close()
            observer.on_complete(stats)


class CleanStage(PipelineStage):
    def process(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        if cfg.dry_run or not ctx.samples:
            return

        cleaner = DatasetCleaner(cfg.cleaner)
        cleaned = cleaner.clean_samples(ctx.samples)
        kept, dropped = cleaner.stats
        ctx.collector.record_cleaner(kept, dropped)
        logger.info(f"Cleaner: kept={kept}, dropped={dropped}")
        ctx.samples = cleaned


class ScoreStage(PipelineStage):
    def process(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        scfg = cfg.scoring
        if not (scfg.enabled and scfg.dimensions) or not ctx.samples:
            return

        scoring_api = create_client(
            model=scfg.model or cfg.api.model,
            api_key=scfg.api_key or cfg.api.api_key,
            base_url=scfg.base_url or cfg.api.base_url,
            retry=scfg.retry or cfg.api.retry,
        )

        scorer = DatasetScorer(scfg)
        ctx.samples = scorer.score_samples(scoring_api, ctx.samples)
        scored, failed = scorer.stats
        ctx.collector.record_scorer(scored, failed)
        ctx.collector.collect_scores(ctx.samples)
        logger.info(f"Scorer: scored={scored}, failed={failed}")


class ScoreFilterStage(PipelineStage):
    def process(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        min_score = cfg.scoring.min_total_score
        if min_score <= 0 or not ctx.samples:
            return

        output_path = cfg.output.path
        kept = dropped = 0

        with open(output_path, "w", encoding="utf-8") as fout:
            for sample in ctx.samples:
                total = sample.get("total_score", 0)
                if total >= min_score:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    dropped += 1

        ctx.collector.record_score_filter(kept, dropped)
        ctx.output_path = output_path
        logger.info(
            f"Score filter (min={min_score}): kept={kept}, dropped={dropped}, result={output_path}"
        )


# Register the built-in pipeline stages.
stage_registry.register("generate", GenerationStage)
stage_registry.register("clean", CleanStage)
stage_registry.register("score", ScoreStage)
stage_registry.register("score_filter", ScoreFilterStage)
