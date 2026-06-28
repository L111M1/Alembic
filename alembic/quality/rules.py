"""Shared quality rules used by both the validator chain and the cleaner.

Previously :class:`LengthValidator` (quality layer) and
``DatasetCleaner._apply_quality`` (cleaner layer) each implemented their own
length / ratio checks against their respective config objects. These are now
expressed once as :class:`QualityRule` strategies and composed into a
:class:`QualityRuleSet` built from whichever config section is relevant.
"""

import abc

from alembic.cleaner.ops import (
    char_repetition_ratio,
    special_char_ratio,
    word_repetition_ratio,
)
from alembic.config import CleanerConfig, QualityConfig


class QualityRule(abc.ABC):
    @abc.abstractmethod
    def check(self, inst: str, out: str) -> bool: ...


class LengthRule(QualityRule):
    def __init__(self, inst_min: int, inst_max: int, out_min: int, out_max: int):
        self._inst_min = inst_min
        self._inst_max = inst_max
        self._out_min = out_min
        self._out_max = out_max

    def check(self, inst: str, out: str) -> bool:
        ilen = len(inst)
        olen = len(out)
        if ilen < self._inst_min or ilen > self._inst_max:
            return False
        if olen < self._out_min or olen > self._out_max:
            return False
        return True


class RatioRule(QualityRule):
    def __init__(
        self,
        max_special_char_ratio: float,
        max_word_repetition_ratio: float,
        max_char_repetition_ratio: float,
    ):
        self._max_special = max_special_char_ratio
        self._max_word_rep = max_word_repetition_ratio
        self._max_char_rep = max_char_repetition_ratio

    def check(self, inst: str, out: str) -> bool:
        if special_char_ratio(inst) > self._max_special:
            return False
        if special_char_ratio(out) > self._max_special:
            return False
        if word_repetition_ratio(out) > self._max_word_rep:
            return False
        if char_repetition_ratio(out) > self._max_char_rep:
            return False
        return True


class QualityRuleSet:
    """A short-circuiting conjunction of :class:`QualityRule` instances."""

    def __init__(self, rules: list[QualityRule]):
        self._rules = rules

    def check(self, inst: str, out: str) -> bool:
        for rule in self._rules:
            if not rule.check(inst, out):
                return False
        return True

    @classmethod
    def for_quality_config(cls, config: QualityConfig) -> "QualityRuleSet":
        return cls([
            LengthRule(
                config.instruction_min_len,
                config.instruction_max_len,
                config.output_min_len,
                config.output_max_len,
            )
        ])

    @classmethod
    def for_cleaner_config(cls, config: CleanerConfig) -> "QualityRuleSet":
        return cls([
            LengthRule(
                config.instruction_min_len,
                config.instruction_max_len,
                config.output_min_len,
                config.output_max_len,
            ),
            RatioRule(
                config.max_special_char_ratio,
                config.max_word_repetition_ratio,
                config.max_char_repetition_ratio,
            ),
        ])
