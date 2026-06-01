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

    def _clean_with_embeddings(self, input_path: str, output_path: str) -> tuple[int, int]:
        candidates = self._load_candidates(input_path)
        if not candidates:
            logger.info(f"Cleaning done: kept=0, dropped={self._dropped_count}")
            return 0, self._dropped_count

        kept = self._semantic_dedup(candidates)

        with open(output_path, "w", encoding="utf-8") as fout:
            for sample in kept:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

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

        result = {"instruction": inst, "output": out}
        if sample.get("system"):
            result["system"] = sample["system"]
        if sample.get("metadata"):
            result["metadata"] = sample["metadata"]
        return result

    def _semantic_dedup(self, candidates: list[dict]) -> list[dict]:
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor, as_completed

        client = EmbeddingClient(
            model=self._config.embedding_model,
            api_key=self._config.embedding_api_key,
            base_url=self._config.embedding_base_url,
        )
        texts = [s["instruction"] + " " + s["output"] for s in candidates]
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

    @property
    def stats(self) -> tuple[int, int]:
        return self._cleaned_count, self._dropped_count
