"""Deduplication strategies for the cleaning pipeline.

The :class:`DedupStrategy` interface lets :class:`DatasetCleaner` treat the
choice of dedup algorithm (exact / minhash / semantic / none) as a pluggable
concern selected at construction time via :func:`build_dedup_strategy`.
"""

import abc
import logging
from typing import Callable, Optional

from alembic.cleaner.ops import compute_dedup_key, minhash_dedup

logger = logging.getLogger(__name__)


def default_sample_text(sample: dict) -> str:
    if "messages" in sample:
        return " ".join(m.get("content", "") for m in sample["messages"])
    return sample.get("instruction", "") + " " + sample.get("output", "")


class DedupStrategy(abc.ABC):
    @abc.abstractmethod
    def filter(self, candidates: list[dict]) -> list[dict]:
        """Return the subset of ``candidates`` that survive deduplication."""


class NoDedup(DedupStrategy):
    def filter(self, candidates: list[dict]) -> list[dict]:
        return candidates


class ExactDedup(DedupStrategy):
    def __init__(self):
        self._seen: set[str] = set()

    def filter(self, candidates: list[dict]) -> list[dict]:
        kept: list[dict] = []
        for sample in candidates:
            key = compute_dedup_key(sample.get("instruction", "") + sample.get("output", ""))
            if key in self._seen:
                continue
            self._seen.add(key)
            kept.append(sample)
        return kept


class MinHashDedup(DedupStrategy):
    def __init__(
        self,
        threshold: float = 0.7,
        num_perm: int = 128,
        ngram_n: int = 3,
        text_fn: Optional[Callable[[dict], str]] = None,
    ):
        self._threshold = threshold
        self._num_perm = num_perm
        self._ngram_n = ngram_n
        self._text_fn = text_fn or default_sample_text

    def filter(self, candidates: list[dict]) -> list[dict]:
        kept, _ = minhash_dedup(
            candidates,
            text_fn=self._text_fn,
            threshold=self._threshold,
            num_perm=self._num_perm,
            ngram_n=self._ngram_n,
        )
        return kept


class SemanticDedup(DedupStrategy):
    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        threshold: float = 0.85,
        batch_size: int = 20,
        text_fn: Optional[Callable[[dict], str]] = None,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._threshold = threshold
        self._batch_size = batch_size
        self._text_fn = text_fn or default_sample_text

    def filter(self, candidates: list[dict]) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import numpy as np

        from alembic.api.embedding import EmbeddingClient

        if not candidates:
            return []

        client = EmbeddingClient(
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
        )
        texts = [self._text_fn(s) for s in candidates]
        batches = [texts[i:i + self._batch_size] for i in range(0, len(texts), self._batch_size)]

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
        for i in range(len(candidates)):
            if not keep_mask[i]:
                continue
            sims = emb_matrix[i] @ emb_matrix[i + 1:].T
            for j in np.where(sims >= self._threshold)[0]:
                keep_mask[i + 1 + j] = False

        return [s for s, m in zip(candidates, keep_mask) if m]


def build_dedup_strategy(config) -> DedupStrategy:
    """Select the dedup strategy based on :class:`CleanerConfig` flags.

    Priority mirrors the original ``DatasetCleaner.clean_file`` branching:
    semantic > minhash > exact > none.
    """
    if config.embedding_dedup:
        return SemanticDedup(
            model=config.embedding_model,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
            threshold=config.embedding_similarity_threshold,
            batch_size=config.embedding_batch_size,
        )
    if config.minhash_dedup:
        return MinHashDedup(
            threshold=config.minhash_threshold,
            num_perm=config.minhash_num_perm,
            ngram_n=config.minhash_ngram_n,
        )
    if config.dedup:
        return ExactDedup()
    return NoDedup()
