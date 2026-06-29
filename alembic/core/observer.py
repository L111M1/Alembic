import abc
import logging
import time
from typing import Optional


class Observer(abc.ABC):
    @abc.abstractmethod
    def on_start(self, total: int) -> None: ...

    @abc.abstractmethod
    def on_sample(self, index: int, success: bool, strategy: str) -> None: ...

    @abc.abstractmethod
    def on_complete(self, stats) -> None: ...


class LogObserver(Observer):
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._log = logger or logging.getLogger(__name__)
        self._start_time: float = 0

    def on_start(self, total: int) -> None:
        self._start_time = time.time()
        self._log.info(f"Starting generation of up to {total} samples")

    def on_sample(self, index: int, success: bool, strategy: str) -> None:
        if success and (index % 50 == 0):
            elapsed = time.time() - self._start_time
            rate = index / elapsed if elapsed > 0 else 0
            self._log.info(f"[{index}] generated | {rate:.1f} samples/s")

    def on_complete(self, stats) -> None:
        elapsed = time.time() - self._start_time
        rate = stats.total_generated / elapsed if elapsed > 0 else 0
        self._log.info(
            f"Done. generated={stats.total_generated} "
            f"filtered={stats.total_filtered} "
            f"elapsed={elapsed:.1f}s "
            f"rate={rate:.1f} samples/s"
        )


class CompositeObserver(Observer):
    def __init__(self, *observers: Observer):
        self._observers = list(observers)

    def add(self, observer: Observer) -> None:
        self._observers.append(observer)

    def on_start(self, total: int) -> None:
        for ob in self._observers:
            ob.on_start(total)

    def on_sample(self, index: int, success: bool, strategy: str) -> None:
        for ob in self._observers:
            ob.on_sample(index, success, strategy)

    def on_complete(self, stats) -> None:
        for ob in self._observers:
            ob.on_complete(stats)
