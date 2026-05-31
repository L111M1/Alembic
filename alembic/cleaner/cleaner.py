import json
import logging
from typing import Iterator, Optional

from alembic.cleaner.ops import (
    clean_text,
    special_char_ratio,
    word_repetition_ratio,
    char_repetition_ratio,
    compute_dedup_key,
)
from alembic.config import CleanerConfig
from alembic.core.types import GenerationSample

logger = logging.getLogger(__name__)


class DatasetCleaner:
    def __init__(self, config: CleanerConfig):
        self._config = config
        self._seen_keys: set[str] = set()
        self._cleaned_count = 0
        self._dropped_count = 0

    def clean_file(self, input_path: str, output_path: str) -> tuple[int, int]:
        with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    self._dropped_count += 1
                    continue

                cleaned = self._clean_sample(sample)
                if cleaned:
                    fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    self._cleaned_count += 1
                else:
                    self._dropped_count += 1

        logger.info(f"Cleaning done: kept={self._cleaned_count}, dropped={self._dropped_count}")
        return self._cleaned_count, self._dropped_count

    def _clean_sample(self, sample: dict) -> Optional[dict]:
        inst = sample.get("instruction", "")
        out = sample.get("output", "") or sample.get("response", "")

        cfg = self._config

        inst = clean_text(inst, cfg.remove_html, cfg.remove_urls, cfg.remove_emails)
        out = clean_text(out, cfg.remove_html, cfg.remove_urls, cfg.remove_emails)

        inst = inst.strip()
        out = out.strip()

        ilen = len(inst)
        olen = len(out)
        if ilen < cfg.instruction_min_len or ilen > cfg.instruction_max_len:
            return None
        if olen < cfg.output_min_len or olen > cfg.output_max_len:
            return None

        if special_char_ratio(inst) > cfg.max_special_char_ratio:
            return None
        if special_char_ratio(out) > cfg.max_special_char_ratio:
            return None

        if word_repetition_ratio(out) > cfg.max_word_repetition_ratio:
            return None

        if char_repetition_ratio(out) > cfg.max_char_repetition_ratio:
            return None

        if cfg.dedup:
            key = compute_dedup_key(inst + out)
            if key in self._seen_keys:
                return None
            self._seen_keys.add(key)

        result = {"instruction": inst, "output": out}
        if sample.get("system"):
            result["system"] = sample["system"]
        if sample.get("metadata"):
            result["metadata"] = sample["metadata"]
        return result

    @property
    def stats(self) -> tuple[int, int]:
        return self._cleaned_count, self._dropped_count
