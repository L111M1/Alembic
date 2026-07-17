import hashlib
import random
import re
from collections import defaultdict

import numpy as np


def clean_text(text: str) -> str:
    return text.strip()


_SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fff\u3400-\u4dbf\s.,!?;:\'"()\-_+=/\\@#$%^&*\[\]{}|<>`~]')


def special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    count = len(_SPECIAL_CHARS_PATTERN.findall(text))
    return count / len(text)


def word_repetition_ratio(text: str, window: int = 5) -> float:
    words = text.lower().split()
    if len(words) < window:
        return 0.0
    seen = set()
    repeat = 0
    for i in range(len(words) - window + 1):
        ngram = tuple(words[i:i + window])
        if ngram in seen:
            repeat += 1
        else:
            seen.add(ngram)
    return repeat / max(len(words) - window + 1, 1)


def char_repetition_ratio(text: str, min_len: int = 5) -> float:
    if len(text) < min_len:
        return 0.0
    max_run = 1
    current_run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    if max_run < min_len:
        return 0.0
    return max_run / len(text)


def ngram_diversity(text: str, n: int = 3, unit: str = "char") -> float:
    """N-gram diversity ratio: unique n-grams / total n-grams.

    A high ratio means the text uses a rich variety of n-grams (good).
    A low ratio means the text is repetitive / templated (bad).
    Returns 1.0 for texts shorter than ``n`` units (trivially diverse).

    Args:
        text: Input text.
        n: N-gram size (default 3, following CCNet / GPT-3 practice).
        unit: ``"char"`` for character n-grams (default, catches phrase
            repetition like "the cat the cat the cat"), or ``"word"`` for
            word n-grams (catches sentence-level repetition).
    """
    text = text.strip()
    if unit == "word":
        tokens = text.lower().split()
        if len(tokens) < n:
            return 1.0
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    else:
        chars = text.lower()
        if len(chars) < n:
            return 1.0
        ngrams = [chars[i:i + n] for i in range(len(chars) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def tokenize_ngrams(text: str, n: int = 3) -> list[str]:
    chars = text.strip().lower()
    if len(chars) < n:
        return [chars] if chars else []
    return [chars[i:i + n] for i in range(len(chars) - n + 1)]


def minhash_signature(tokens: list[str], num_perm: int = 128, seed: int = 42) -> list[int]:
    """Fast MinHash signature via universal hashing.

    For each unique token we call SHA256 *once*, cache the result, then
    apply random linear functions (a * h + b) % M per permutation.
    Reduces hashing from O(num_perm * len(tokens)) to O(len(unique_tokens)).
    """
    if not tokens:
        return [0] * num_perm
    M = (1 << 61) - 1  # large Mersenne prime
    # one SHA per unique token
    token_hashes: dict[str, int] = {}
    for t in set(tokens):
        d = hashlib.sha256(f"{seed}:{t}".encode("utf-8")).digest()
        token_hashes[t] = int.from_bytes(d[:8], "big")
    rng = random.Random(seed)
    coeffs = [(rng.randrange(1, M - 1), rng.randrange(0, M - 1)) for _ in range(num_perm)]
    sig = [M] * num_perm
    for t in tokens:
        hv = token_hashes[t]
        for i, (a, b) in enumerate(coeffs):
            h = (a * hv + b) % M
            if h < sig[i]:
                sig[i] = h
    return sig


def minhash_similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b) or len(sig_a) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def minhash_bruteforce_dedup(
    samples: list[dict],
    text_fn,
    threshold: float = 0.7,
    num_perm: int = 128,
    ngram_n: int = 3,
) -> tuple[list[dict], list[int]]:
    """O(n²) brute-force MinHash dedup — suitable for <50K samples."""
    texts = [text_fn(s) for s in samples]
    signatures = [minhash_signature(tokenize_ngrams(t, ngram_n), num_perm) for t in texts]

    keep_mask = [True] * len(samples)
    sign_arr = np.array(signatures, dtype=np.uint64)

    for i in range(len(samples)):
        if not keep_mask[i]:
            continue
        matches = np.sum(sign_arr[i] == sign_arr[i + 1:], axis=1)
        sims = matches / num_perm
        for j in np.where(sims >= threshold)[0]:
            keep_mask[i + 1 + j] = False

    kept = [s for s, m in zip(samples, keep_mask) if m]
    dropped_indices = [idx for idx, m in enumerate(keep_mask) if not m]
    return kept, dropped_indices


def minhash_lsh_dedup(
    samples: list[dict],
    text_fn,
    threshold: float = 0.7,
    num_perm: int = 128,
    ngram_n: int = 3,
    bands: int = 20,
) -> tuple[list[dict], list[int]]:
    """LSH-accelerated MinHash dedup — O(n²) → O(n) candidates.

    Divides the ``num_perm``-dimensional signature into ``bands`` bands;
    documents that share *any* band hash become candidates for full comparison.
    """
    texts = [text_fn(s) for s in samples]
    signatures = [minhash_signature(tokenize_ngrams(t, ngram_n), num_perm) for t in texts]

    rows_per_band = num_perm // bands
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for doc_id, sig in enumerate(signatures):
        for b in range(bands):
            start = b * rows_per_band
            band = tuple(sig[start:start + rows_per_band])
            buckets[(b, band)].append(doc_id)

    keep_mask = [True] * len(samples)
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            if not keep_mask[ids[i]]:
                continue
            for j in range(i + 1, len(ids)):
                if not keep_mask[ids[j]]:
                    continue
                sim = minhash_similarity(signatures[ids[i]], signatures[ids[j]])
                if sim >= threshold:
                    keep_mask[ids[j]] = False

    kept = [s for s, m in zip(samples, keep_mask) if m]
    dropped_indices = [idx for idx, m in enumerate(keep_mask) if not m]
    return kept, dropped_indices

