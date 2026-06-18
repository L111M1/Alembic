import logging
import random
from typing import Iterator, Optional

from alembic.api.base import BaseAPIClient
from alembic.core.types import GenerationSample
from alembic.strategies.base import GenerationStrategy
from alembic.strategies.seed_driven import SeedDrivenStrategy
from alembic.strategies.self_instruct import SelfInstructStrategy
from alembic.strategies.topic_driven import TopicDrivenStrategy

logger = logging.getLogger(__name__)

_STRATEGY_MAP = {
    "topic_driven": TopicDrivenStrategy,
    "seed_driven": SeedDrivenStrategy,
    "self_instruct": SelfInstructStrategy,
}


class CompositeStrategy(GenerationStrategy):
    def __init__(self, api: BaseAPIClient, strategy_configs: list):
        super().__init__(api, {})
        self._strategies: list[tuple[GenerationStrategy, float]] = []
        for cfg in strategy_configs:
            stype = cfg.get("type", cfg.get("_type", ""))
            weight = float(cfg.get("weight", 1.0))
            params = {k: v for k, v in cfg.items() if k not in ("type", "weight")}
            strategy = _create_strategy(stype, api, params)
            if strategy:
                self._strategies.append((strategy, weight))
        if not self._strategies:
            raise ValueError("No valid strategies configured")

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        for strategy, _ in self._strategies:
            yield from strategy.iter_prompts()

    def generate(self) -> Iterator[GenerationSample]:
        for strategy, _ in self._strategies:
            yield from strategy.generate()

    def estimated_count(self) -> int:
        return sum(s.estimated_count() for s, _ in self._strategies)

    def _weighted_choice(self) -> GenerationStrategy:
        total = sum(w for _, w in self._strategies) or 1.0
        r = random.random() * total
        cum = 0.0
        for s, w in self._strategies:
            cum += w
            if r < cum:
                return s
        return self._strategies[0][0]


def create_strategy(api: BaseAPIClient, strategy_cfgs: list) -> GenerationStrategy:
    if len(strategy_cfgs) == 1:
        cfg = strategy_cfgs[0]
        stype = cfg.get("type", cfg.get("_type", ""))
        params = {k: v for k, v in cfg.items() if k not in ("type", "weight")}
        return _create_strategy(stype, api, params)
    return CompositeStrategy(api, strategy_cfgs)


def _create_strategy(stype: str, api: BaseAPIClient, params: dict) -> Optional[GenerationStrategy]:
    cls = _STRATEGY_MAP.get(stype)
    if cls is None:
        logger.warning(f"Unknown strategy type '{stype}', skipping")
        return None
    return cls(api, params)
