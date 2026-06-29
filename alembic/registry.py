import logging
from typing import Generic, Optional, Type, TypeVar, Union

from alembic.api.base import BaseAPIClient, RetryConfig, with_retry
from alembic.api.providers import OpenAICompatibleClient
from alembic.strategies.base import GenerationStrategy
from alembic.strategies.composite import CompositeStrategy
from alembic.strategies.seed_driven import SeedDrivenStrategy
from alembic.strategies.self_instruct import SelfInstructStrategy
from alembic.strategies.topic_driven import TopicDrivenStrategy

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── generic base ──────────────────────────────────────────────────────────

class Registry(Generic[T]):
    """Generic registry mapping a name to a registered value (class, factory, etc.).

    Used by :class:`ProviderRegistry`, :class:`StrategyRegistry`, and
    :class:`StageRegistry` to avoid duplicating the same dict-wrapper pattern.
    """

    def __init__(self):
        self._map: dict[str, T] = {}

    def register(self, name: str, value: T) -> None:
        self._map[name] = value

    def get(self, name: str) -> Union[T, None]:
        return self._map.get(name)

    def names(self) -> list[str]:
        return list(self._map)


# ── API provider registry ─────────────────────────────────────────────────

class ProviderRegistry(Registry[Type[BaseAPIClient]]):
    """Registry mapping a provider name to a :class:`BaseAPIClient` subclass.

    New providers can be plugged in without touching the factory:
    ``provider_registry.register("anthropic", AnthropicClient)``.
    """

    pass


provider_registry = ProviderRegistry()
provider_registry.register("openai", OpenAICompatibleClient)


def create_client(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    retry: Optional[dict] = None,
    provider: str = "openai",
    **kwargs,
) -> BaseAPIClient:
    cls = provider_registry.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider '{provider}'")
    client = cls(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )

    if retry:
        rc = RetryConfig(
            max_retries=retry.get("max_retries", 3),
            initial_delay=retry.get("initial_delay", 1.0),
            backoff_multiplier=retry.get("backoff_multiplier", 2.0),
            max_delay=retry.get("max_delay", 30.0),
        )
        client = with_retry(client, rc)

    logger.info(f"Created client (provider={provider}, model={model})")
    return client


# ── strategy registry ─────────────────────────────────────────────────────

class StrategyRegistry(Registry[type[GenerationStrategy]]):
    """Registry mapping a strategy type name to its class.

    Allows new strategies to be registered without modifying the factory:
    ``strategy_registry.register("my_strategy", MyStrategy)``.
    """

    def create(
        self, name: str, api: BaseAPIClient, params: dict
    ) -> Optional[GenerationStrategy]:
        cls = super().get(name)
        if cls is None:
            logger.warning(f"Unknown strategy type '{name}', skipping")
            return None
        return cls(api, params)


strategy_registry = StrategyRegistry()
strategy_registry.register("topic_driven", TopicDrivenStrategy)
strategy_registry.register("seed_driven", SeedDrivenStrategy)
strategy_registry.register("self_instruct", SelfInstructStrategy)


# Backward-compatible alias.
def _create_strategy(
    stype: str, api: BaseAPIClient, params: dict
) -> Optional[GenerationStrategy]:
    return strategy_registry.create(stype, api, params)


# ── stage registry ────────────────────────────────────────────────────────

class StageRegistry(Registry[type]):
    """Registry mapping a stage name to a :class:`PipelineStage` subclass.

    New stages can be plugged in:
    ``stage_registry.register("translate", TranslateStage)``.
    The default built-in stages are registered in :mod:`alembic.core.stages`.
    """

    def create(self, name: str, **kwargs):
        cls = super().get(name)
        if cls is None:
            raise ValueError(f"Unknown stage type '{name}'")
        return cls(**kwargs)


stage_registry = StageRegistry()


# ── composite strategy factory ────────────────────────────────────────────

def create_strategy(api: BaseAPIClient, strategy_cfgs: list) -> GenerationStrategy:
    if len(strategy_cfgs) == 1:
        cfg = strategy_cfgs[0]
        stype = cfg.get("type", cfg.get("_type", ""))
        params = {k: v for k, v in cfg.items() if k not in ("type", "weight")}
        strategy = strategy_registry.create(stype, api, params)
        if strategy is None:
            raise ValueError(f"Unknown strategy type '{stype}'")
        return strategy

    children: list[tuple[GenerationStrategy, float]] = []
    for cfg in strategy_cfgs:
        stype = cfg.get("type", cfg.get("_type", ""))
        weight = float(cfg.get("weight", 1.0))
        params = {k: v for k, v in cfg.items() if k not in ("type", "weight")}
        strategy = strategy_registry.create(stype, api, params)
        if strategy:
            children.append((strategy, weight))
    return CompositeStrategy(api, children)
