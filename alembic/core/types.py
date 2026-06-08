from dataclasses import dataclass, field


@dataclass
class SeedSample:
    instruction: str = ""
    output: str = ""
    system: str = ""
    messages: list[dict] = field(default_factory=list)


@dataclass
class GenerationSample:
    instruction: str = ""
    output: str = ""
    system: str = ""
    messages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_multi_turn(self) -> bool:
        return len(self.messages) > 0


@dataclass
class GenerationStats:
    total_attempted: int = 0
    total_generated: int = 0
    total_filtered: int = 0
    by_strategy: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


@dataclass
class PipelineConfig:
    strategies: list = field(default_factory=list)
    api: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
