import abc
import json
import logging
from dataclasses import dataclass, field
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


@dataclass
class PipelineContext:
    """Mutable carrier passed along the pipeline stages."""
    config: AppConfig
    collector: StatisticsCollector
    stats: GenerationStats = field(default_factory=GenerationStats)
    output_path: str = ""
    cleaned_path: str = ""
    scored_path: str = ""


class PipelineStage(abc.ABC):
    """A single stage in the generate -> clean -> score -> filter pipeline."""

    @abc.abstractmethod
    def process(self, ctx: PipelineContext) -> None: ...


def _derive_path(base: str, suffix: str) -> str:
    if base.endswith(f"_{suffix}.jsonl"):
        return base
    return base.replace(".jsonl", f"_{suffix}.jsonl")


class GenerationStage(PipelineStage):
    def process(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        api = create_client(
            model=cfg.api.model,
            api_key=cfg.api.api_key,
            base_url=cfg.api.base_url,
            retry=cfg.api.retry,
        )

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
            strategy_cfgs.append(d)
        strategy = create_strategy(api, strategy_cfgs)

        validator = build_validator_chain(cfg.quality)
        writer = None if cfg.dry_run else JSONLWriter(cfg.output)
        observer = CompositeObserver(LogObserver(logger), ctx.collector)

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

                if validator.validate(sample):
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

        ctx.output_path = cfg.output.path


class CleanStage(PipelineStage):
    def process(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        if cfg.dry_run or not cfg.output.path:
            return

        output_path = cfg.output.path
        if not Path(output_path).exists():
            ctx.cleaned_path = output_path
            return

        cleaner = DatasetCleaner(cfg.cleaner)
        cleaned_path = _derive_path(output_path, "cleaned")

        kept, dropped = cleaner.clean_file(output_path, cleaned_path)
        ctx.collector.record_cleaner(kept, dropped)
        logger.info(f"Cleaner: kept={kept}, dropped={dropped}, result={cleaned_path}")
        ctx.cleaned_path = cleaned_path


class ScoreStage(PipelineStage):
    def process(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        scfg = cfg.scoring
        if not (scfg.enabled and scfg.dimensions) or not ctx.cleaned_path:
            return

        input_path = ctx.cleaned_path
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
        ctx.collector.record_scorer(scored, failed)
        ctx.collector.record_scores(output_path)
        logger.info(f"Scorer: scored={scored}, failed={failed}, result={output_path}")
        ctx.scored_path = output_path


class ScoreFilterStage(PipelineStage):
    def process(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        min_score = cfg.scoring.min_total_score
        if min_score <= 0 or not ctx.scored_path:
            return

        input_path = ctx.scored_path
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
                    fout.write(
                        json.dumps(
                            {
                                "instruction": sample.get("instruction", ""),
                                "output": sample.get("output", ""),
                                "metadata": sample.get("metadata", {}),
                                "scores": sample.get("scores", {}),
                                "total_score": total,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    kept += 1
                else:
                    dropped += 1

        ctx.collector.record_score_filter(kept, dropped)
        logger.info(
            f"Score filter (min={min_score}): kept={kept}, dropped={dropped}, result={output_path}"
        )
