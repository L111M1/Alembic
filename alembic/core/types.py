import random
from dataclasses import dataclass, field


DEFAULT_TOPICS = {
    "cs": {"en": "Computer Science", "zh": "计算机科学"},
    "ai": {"en": "Artificial Intelligence", "zh": "人工智能"},
    "math": {"en": "Mathematics", "zh": "数学"},
    "science": {"en": "Natural Science", "zh": "自然科学"},
    "engineering": {"en": "Engineering", "zh": "工程技术"},
    "social": {"en": "Social Science", "zh": "社会科学"},
    "humanities": {"en": "Humanities & Arts", "zh": "人文艺术"},
    "health": {"en": "Medicine & Health", "zh": "医学健康"},
    "business": {"en": "Business Management", "zh": "商业管理"},
    "education": {"en": "Education", "zh": "教育学习"},
}


def random_topic(lang: str = "en") -> str:
    """Pick a random default topic as fallback when none is specified."""
    key = random.choice(list(DEFAULT_TOPICS))
    return DEFAULT_TOPICS[key].get(lang, DEFAULT_TOPICS[key]["en"])


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
