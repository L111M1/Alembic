import json
import logging
import random
from pathlib import Path

from alembic.config import AppConfig
from alembic.core.stages import PipelineContext, PipelineStage
from alembic.core.stats import StatisticsCollector
from alembic.core.types import GenerationStats
from alembic.registry import stage_registry

logger = logging.getLogger(__name__)

class Pipeline:
    """Facade orchestrating a chain of :class:`PipelineStage` instances.

    The generate -> clean -> score -> filter flow is composed from independent
    stage objects registered in :data:`stage_registry`.  Custom stages can be
    added to the registry before the pipeline is constructed.
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._collector = StatisticsCollector()
        self._stage_names: list[str] = ["generate", "clean", "score", "score_filter"]

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        return cls(AppConfig.from_yaml(path))

    def set_stages(self, *names: str) -> "Pipeline":
        self._stage_names = list(names)
        return self

    def _build_stages(self) -> list[PipelineStage]:
        return [stage_registry.create(name) for name in self._stage_names]

    def run(self, profile: bool = False) -> GenerationStats:
        if self._config.random_seed is not None:
            random.seed(self._config.random_seed)

        ctx = PipelineContext(config=self._config, collector=self._collector)
        for stage in self._build_stages():
            name = type(stage).__name__
            try:
                stage.process(ctx)
            except Exception as exc:
                logger.error("Stage %s failed, skipping: %s", name, exc, exc_info=True)

        if not self._config.dry_run and ctx.samples and self._config.output.path:
            out_path = Path(self._config.output.path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for s in ctx.samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            logger.info("Wrote %d samples to %s", len(ctx.samples), out_path)
            self._collector.save_report(self._config.output.path)

        return ctx.stats
