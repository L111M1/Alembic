import logging
import random
from queue import Queue
from threading import Thread
from typing import Iterator, Optional, TypeVar

from alembic.api.base import BaseAPIClient
from alembic.core.types import GenerationSample
from alembic.strategies.base import GenerationStrategy
from alembic.strategies.seed_driven import SeedDrivenStrategy
from alembic.strategies.self_instruct import SelfInstructStrategy
from alembic.strategies.topic_driven import TopicDrivenStrategy

logger = logging.getLogger(__name__)

T = TypeVar("T")


def merge_generators(generators: list[Iterator[T]]) -> Iterator[T]:
    """Interleave multiple iterators using one daemon thread per source.

    The single-source fast path simply delegates with no threading overhead.
    This is the single implementation of the producer/consumer merge that was
    previously duplicated between ``iter_prompts`` and ``generate``.
    """
    if len(generators) <= 1:
        for gen in generators:
            yield from gen
        return

    q: Queue = Queue()

    def produce(gen: Iterator[T]) -> None:
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(None)

    threads = [Thread(target=produce, args=(g,), daemon=True) for g in generators]
    for t in threads:
        t.start()

    done = 0
    while done < len(threads):
        item = q.get()
        if item is None:
            done += 1
        else:
            yield item

    for t in threads:
        t.join()


class StrategyRegistry:
    """Registry mapping a strategy type name to its class.

    Allows new strategies to be registered without modifying the factory:
    ``StrategyRegistry.register("my_strategy", MyStrategy)``.
    """

    def __init__(self):
        self._map: dict[str, type[GenerationStrategy]] = {}

    def register(self, name: str, cls: type[GenerationStrategy]) -> None:
        self._map[name] = cls

    def create(
        self, name: str, api: BaseAPIClient, params: dict
    ) -> Optional[GenerationStrategy]:
        cls = self._map.get(name)
        if cls is None:
            logger.warning(f"Unknown strategy type '{name}', skipping")
            return None
        return cls(api, params)

    def names(self) -> list[str]:
        return list(self._map)


# Default registry with the built-in strategies pre-registered.
registry = StrategyRegistry()
registry.register("topic_driven", TopicDrivenStrategy)
registry.register("seed_driven", SeedDrivenStrategy)
registry.register("self_instruct", SelfInstructStrategy)


class CompositeStrategy(GenerationStrategy):
    def __init__(self, api: BaseAPIClient, strategy_configs: list):
        super().__init__(api, {})
        self._strategies: list[tuple[GenerationStrategy, float]] = []
        for cfg in strategy_configs:
            stype = cfg.get("type", cfg.get("_type", ""))
            weight = float(cfg.get("weight", 1.0))
            params = {k: v for k, v in cfg.items() if k not in ("type", "weight")}
            strategy = registry.create(stype, api, params)
            if strategy:
                self._strategies.append((strategy, weight))
        if not self._strategies:
            raise ValueError("No valid strategies configured")

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        return merge_generators([s.iter_prompts() for s, _ in self._strategies])

    def generate(self) -> Iterator[GenerationSample]:
        return merge_generators([s.generate() for s, _ in self._strategies])

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
        strategy = registry.create(stype, api, params)
        if strategy is None:
            raise ValueError(f"Unknown strategy type '{stype}'")
        return strategy
    return CompositeStrategy(api, strategy_cfgs)


# Backward-compatible alias for the previous private factory function.
def _create_strategy(
    stype: str, api: BaseAPIClient, params: dict
) -> Optional[GenerationStrategy]:
    return registry.create(stype, api, params)
