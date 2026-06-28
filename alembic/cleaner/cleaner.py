import json
import logging
from typing import Optional

from alembic.cleaner.dedup import DedupStrategy, build_dedup_strategy
from alembic.cleaner.ops import clean_text
from alembic.config import CleanerConfig
from alembic.quality.rules import QualityRuleSet

logger = logging.getLogger(__name__)


class DatasetCleaner:
    """Cleans a JSONL dataset using a Template Method over a pluggable dedup strategy.

    The cleaning flow is fixed (normalize -> quality filter -> dedup -> format),
    while the dedup algorithm is supplied by a :class:`DedupStrategy` instance
    selected from the config via :func:`build_dedup_strategy`.
    """

    def __init__(self, config: CleanerConfig, dedup: Optional[DedupStrategy] = None):
        self._config = config
        self._dedup = dedup if dedup is not None else build_dedup_strategy(config)
        self._rules = QualityRuleSet.for_cleaner_config(config)
        self._cleaned_count = 0
        self._dropped_count = 0

    def clean_file(self, input_path: str, output_path: str) -> tuple[int, int]:
        candidates = self._load_candidates(input_path)

        kept = self._dedup.filter(candidates)
        self._cleaned_count = len(kept)
        self._dropped_count += len(candidates) - len(kept)

        self._write(kept, output_path)
        logger.info(f"Cleaning done: kept={self._cleaned_count}, dropped={self._dropped_count}")
        return self._cleaned_count, self._dropped_count

    def _load_candidates(self, input_path: str) -> list[dict]:
        candidates: list[dict] = []
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    self._dropped_count += 1
                    continue
                cleaned = self._basic_clean(sample)
                if cleaned:
                    candidates.append(cleaned)
                else:
                    self._dropped_count += 1
        return candidates

    def _normalize(self, sample: dict) -> Optional[dict]:
        cfg = self._config

        if cfg.field_map:
            sample = {v: sample.get(k, "") for k, v in cfg.field_map.items()}

        fmt = cfg.input_format
        if fmt == "alpaca" and "messages" in sample and isinstance(sample["messages"], list):
            fmt = "chatml"

        if fmt == "chatml":
            messages = sample.get("messages", [])
            system_text = ""
            user_parts = []
            assistant_parts = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "") or ""
                if role == "system":
                    system_text = content
                elif role == "user":
                    if content:
                        user_parts.append(content)
                elif role == "assistant":
                    if content:
                        assistant_parts.append(content)
                    for tc in m.get("tool_calls", []):
                        fn = tc.get("function", {})
                        assistant_parts.append(fn.get("name", ""))
                        assistant_parts.append(fn.get("arguments", ""))
            if not user_parts and not assistant_parts:
                return None
            return {
                "instruction": "\n".join(user_parts),
                "output": "\n".join(assistant_parts),
                "system": system_text,
                "metadata": sample.get("metadata"),
                "_raw": sample,
            }
        else:
            instruction = sample.get("instruction", "")
            output = sample.get("output", "") or sample.get("response", "")
            return {
                "instruction": instruction,
                "output": output,
                "system": sample.get("system", ""),
                "metadata": sample.get("metadata"),
                "_raw": sample,
            }

    def _apply_quality(self, inst: str, out: str):
        cfg = self._config

        inst = clean_text(inst, cfg.remove_html, cfg.remove_urls, cfg.remove_emails).strip()
        out = clean_text(out, cfg.remove_html, cfg.remove_urls, cfg.remove_emails).strip()

        passed = self._rules.check(inst, out)
        return passed, inst, out

    def _format_output(self, normalized: dict, inst: str, out: str) -> dict:
        cfg = self._config
        raw = normalized["_raw"]

        if "messages" in raw and isinstance(raw["messages"], list):
            messages = raw["messages"]
            cleaned_msgs = []
            for m in messages:
                msg = dict(m)
                content = msg.get("content")
                if content is not None and isinstance(content, str):
                    msg["content"] = clean_text(
                        content, cfg.remove_html, cfg.remove_urls, cfg.remove_emails
                    ).strip()
                cleaned_msgs.append(msg)
            result = {"messages": cleaned_msgs}
            if raw.get("system"):
                result["system"] = raw["system"]
            if raw.get("metadata"):
                result["metadata"] = raw["metadata"]
            return result
        else:
            result = {"instruction": inst, "output": out}
            if normalized.get("system"):
                result["system"] = normalized["system"]
            if normalized.get("metadata"):
                result["metadata"] = normalized["metadata"]
            return result

    def _basic_clean(self, sample: dict) -> Optional[dict]:
        normalized = self._normalize(sample)
        if normalized is None:
            return None

        passed, inst, out = self._apply_quality(normalized["instruction"], normalized["output"])
        if not passed:
            return None

        result = {"instruction": inst, "output": out}
        if normalized.get("system"):
            result["system"] = normalized["system"]
        if normalized.get("metadata"):
            result["metadata"] = normalized["metadata"]
        result["_raw"] = normalized.get("_raw")
        return result

    def _write(self, kept: list[dict], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as fout:
            for sample in kept:
                raw = sample.pop("_raw", None)
                if raw and "messages" in raw and isinstance(raw["messages"], list):
                    record = self._format_output({"_raw": raw}, sample["instruction"], sample["output"])
                else:
                    record = sample
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def stats(self) -> tuple[int, int]:
        return self._cleaned_count, self._dropped_count
