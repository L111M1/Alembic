import logging
import random

from alembic.config import AppConfig
from alembic.core.stages import (
    CleanStage,
    GenerationStage,
    PipelineContext,
    PipelineStage,
    ScoreFilterStage,
    ScoreStage,
)
from alembic.core.stats import StatisticsCollector
from alembic.core.types import GenerationStats

logger = logging.getLogger(__name__)


class Pipeline:
    """Facade orchestrating a chain of :class:`PipelineStage` instances.

    The generate -> clean -> score -> filter flow is composed from independent
    stage objects; this class only wires them together and owns the shared
    :class:`PipelineContext` and :class:`StatisticsCollector`.
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._collector = StatisticsCollector()

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        return cls(AppConfig.from_yaml(path))

    def _build_stages(self) -> list[PipelineStage]:
        return [
            GenerationStage(),
            CleanStage(),
            ScoreStage(),
            ScoreFilterStage(),
        ]

    def run(self) -> GenerationStats:
        if self._config.random_seed is not None:
            random.seed(self._config.random_seed)

        ctx = PipelineContext(config=self._config, collector=self._collector)
        for stage in self._build_stages():
            stage.process(ctx)

        if not self._config.dry_run and self._config.output.path:
            self._collector.save_report(self._config.output.path)

        return ctx.stats
