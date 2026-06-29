import logging
from queue import Queue
from threading import Thread
from typing import Iterator, TypeVar

from alembic.api.base import BaseAPIClient
from alembic.core.types import GenerationSample
from alembic.strategies.base import GenerationStrategy

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


class CompositeStrategy(GenerationStrategy):
    def __init__(self, api: BaseAPIClient, strategies: list[tuple[GenerationStrategy, float]]):
        super().__init__(api, {})
        self._strategies = strategies
        if not self._strategies:
            raise ValueError("No valid strategies configured")

    def iter_prompts(self) -> Iterator[tuple[str, list[dict]]]:
        return merge_generators([s.iter_prompts() for s, _ in self._strategies])

    def generate(self) -> Iterator[GenerationSample]:
        return merge_generators([s.generate() for s, _ in self._strategies])

    def estimated_count(self) -> int:
        return sum(s.estimated_count() for s, _ in self._strategies)
