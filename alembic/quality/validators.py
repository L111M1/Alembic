from __future__ import annotations

import abc
import hashlib
import logging
from typing import Optional

from alembic.config import QualityConfig
from alembic.core.types import GenerationSample
from alembic.quality.rules import QualityRuleSet

logger = logging.getLogger(__name__)


class QualityValidator(abc.ABC):
    def __init__(self):
        self._next: Optional[QualityValidator] = None

    def set_next(self, validator: QualityValidator) -> QualityValidator:
        self._next = validator
        return validator

    def validate(self, sample: GenerationSample) -> bool:
        if self._next is None:
            return self._do_validate(sample)
        if self._do_validate(sample):
            return self._next.validate(sample)
        return False

    @abc.abstractmethod
    def _do_validate(self, sample: GenerationSample) -> bool: ...


def _extract_inst_out(sample: GenerationSample) -> tuple[str, str]:
    if sample.is_multi_turn:
        inst = " ".join(m["content"] for m in sample.messages if m.get("role") == "user")
        out = " ".join(m["content"] for m in sample.messages if m.get("role") == "assistant")
    else:
        inst = sample.instruction
        out = sample.output
    return inst, out


class LengthValidator(QualityValidator):
    def __init__(self, config: QualityConfig):
        super().__init__()
        self._rules = QualityRuleSet.for_quality_config(config)

    def _do_validate(self, sample: GenerationSample) -> bool:
        inst, out = _extract_inst_out(sample)
        return self._rules.check(inst, out)


class DedupValidator(QualityValidator):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self._enabled = enabled
        self._seen: set[str] = set()

    def _do_validate(self, sample: GenerationSample) -> bool:
        if not self._enabled:
            return True
        if sample.is_multi_turn:
            text = " ".join(m["content"].strip().lower() for m in sample.messages)
        else:
            text = sample.instruction.strip().lower() + sample.output.strip().lower()
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def build_validator_chain(config: QualityConfig) -> QualityValidator:
    length = LengthValidator(config)
    dedup = DedupValidator(config.dedup)
    length.set_next(dedup)
    return length
