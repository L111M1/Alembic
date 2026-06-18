import json
import logging
from typing import Optional

from alembic.api.embedding import EmbeddingClient
from alembic.cleaner.ops import (
    char_repetition_ratio,
    clean_text,
    compute_dedup_key,
    special_char_ratio,
    word_repetition_ratio,
)
from alembic.config import CleanerConfig

logger = logging.getLogger(__name__)


class DatasetCleaner:
    def __init__(self, config: CleanerConfig):
        self._config = config
        self._seen_keys: set[str] = set()
        self._cleaned_count = 0
        self._dropped_count = 0

    def clean_file(self, input_path: str, output_path: str) -> tuple[int, int]:
        if self._config.embedding_dedup:
            return self._clean_with_embeddings(input_path, output_path)

        with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
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

        ilen = len(inst)
        olen = len(out)
        if ilen < cfg.instruction_min_len or ilen > cfg.instruction_max_len:
            return False, inst, out
        if olen < cfg.output_min_len or olen > cfg.output_max_len:
            return False, inst, out

        if special_char_ratio(inst) > cfg.max_special_char_ratio:
            return False, inst, out
        if special_char_ratio(out) > cfg.max_special_char_ratio:
            return False, inst, out

        if word_repetition_ratio(out) > cfg.max_word_repetition_ratio:
            return False, inst, out

        if char_repetition_ratio(out) > cfg.max_char_repetition_ratio:
            return False, inst, out

        return True, inst, out

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

    def _clean_sample(self, sample: dict) -> Optional[dict]:
        normalized = self._normalize(sample)
        if normalized is None:
            return None

        passed, inst, out = self._apply_quality(normalized["instruction"], normalized["output"])
        if not passed:
            return None

        if self._config.dedup:
            key = compute_dedup_key(inst + out)
            if key in self._seen_keys:
                return None
            self._seen_keys.add(key)

        return self._format_output(normalized, inst, out)

    def _clean_with_embeddings(self, input_path: str, output_path: str) -> tuple[int, int]:
        candidates = self._load_candidates(input_path)
        if not candidates:
            logger.info(f"Cleaning done: kept=0, dropped={self._dropped_count}")
            return 0, self._dropped_count

        kept = self._semantic_dedup(candidates)

        with open(output_path, "w", encoding="utf-8") as fout:
            for sample in kept:
                raw = sample.pop("_raw", None)
                if raw and "messages" in raw and isinstance(raw["messages"], list):
                    record = self._format_output({"_raw": raw}, sample["instruction"], sample["output"])
                else:
                    record = sample
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._cleaned_count = len(kept)
        self._dropped_count += len(candidates) - len(kept)
        logger.info(f"Cleaning done: kept={self._cleaned_count}, dropped={self._dropped_count}")
        return self._cleaned_count, self._dropped_count

    def _load_candidates(self, input_path: str) -> list[dict]:
        candidates = []
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

    def _semantic_dedup(self, candidates: list[dict]) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import numpy as np

        client = EmbeddingClient(
            model=self._config.embedding_model,
            api_key=self._config.embedding_api_key,
            base_url=self._config.embedding_base_url,
        )
        texts = [self._sample_text(s) for s in candidates]
        batch_size = self._config.embedding_batch_size
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        all_embeddings: list[list[float]] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=min(10, len(batches))) as executor:
            futures = {executor.submit(client.embed, batch): i for i, batch in enumerate(batches)}
            for future in as_completed(futures):
                idx = futures[future]
                all_embeddings[idx] = future.result()

        all_embeddings = [e for emb in all_embeddings for e in emb]

        emb_matrix = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_matrix = emb_matrix / norms

        keep_mask = [True] * len(candidates)
        threshold = self._config.embedding_similarity_threshold

        for i in range(len(candidates)):
            if not keep_mask[i]:
                continue
            sims = emb_matrix[i] @ emb_matrix[i + 1:].T
            for j in np.where(sims >= threshold)[0]:
                keep_mask[i + 1 + j] = False

        kept = [s for s, m in zip(candidates, keep_mask) if m]
        return kept

    def _sample_text(self, sample: dict) -> str:
        if "messages" in sample:
            return " ".join(m.get("content", "") for m in sample["messages"])
        return sample.get("instruction", "") + " " + sample.get("output", "")

    @property
    def stats(self) -> tuple[int, int]:
        return self._cleaned_count, self._dropped_count
