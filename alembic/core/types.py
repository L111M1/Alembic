import random
from dataclasses import dataclass, field


DEFAULT_TOPICS = {
    "cs": "计算机科学",
    "ai": "人工智能",
    "math": "数学",
    "science": "自然科学",
    "engineering": "工程技术",
    "social": "社会科学",
    "humanities": "人文艺术",
    "health": "医学健康",
    "business": "商业管理",
    "education": "教育学习",
}


def random_topic() -> str:
    """Pick a random default topic as fallback when none is specified."""
    return random.choice(list(DEFAULT_TOPICS))


@dataclass
class SeedSample:
    instruction: str = ""
    output: str = ""
    system: str = ""
    topic: str = ""
    messages: list[dict] = field(default_factory=list)


@dataclass
class GenerationSample:
    instruction: str = ""
    output: str = ""
    reasoning: str = ""
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
