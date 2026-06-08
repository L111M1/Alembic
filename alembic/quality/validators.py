from __future__ import annotations
import abc
import hashlib
import logging
from typing import Optional

from alembic.core.types import GenerationSample
from alembic.config import QualityConfig

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


class LengthValidator(QualityValidator):
    def __init__(self, config: QualityConfig):
        super().__init__()
        self._inst_min = config.instruction_min_len
        self._inst_max = config.instruction_max_len
        self._out_min = config.output_min_len
        self._out_max = config.output_max_len

    def _do_validate(self, sample: GenerationSample) -> bool:
        if sample.is_multi_turn:
            inst = " ".join(m["content"] for m in sample.messages if m.get("role") == "user")
            out = " ".join(m["content"] for m in sample.messages if m.get("role") == "assistant")
        else:
            inst = sample.instruction
            out = sample.output
        ilen = len(inst)
        olen = len(out)
        if ilen < self._inst_min or ilen > self._inst_max:
            return False
        if olen < self._out_min or olen > self._out_max:
            return False
        return True


class TruncationValidator(QualityValidator):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self._enabled = enabled

    def _do_validate(self, sample: GenerationSample) -> bool:
        if not self._enabled:
            return True
        if sample.is_multi_turn:
            out_texts = [m["content"] for m in sample.messages if m.get("role") == "assistant"]
            output = " ".join(out_texts) if out_texts else ""
        else:
            output = sample.output
        output = output.strip().rstrip('"').rstrip("'").rstrip("`")
        if len(output) < 10:
            return False
        if output.endswith((".", "!", "?", ")", "]", "\n")):
            return True
        last_block = output.split("\n")[-1].strip()
        if len(last_block) < 5:
            return False
        return True


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
    chain = LengthValidator(config)
    chain.set_next(TruncationValidator(config.remove_truncated))
    chain.set_next(DedupValidator(config.dedup))
    return chain
