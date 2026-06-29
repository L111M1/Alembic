import hashlib
import re


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


def tokenize_ngrams(text: str, n: int = 3) -> list[str]:
    chars = text.strip().lower()
    if len(chars) < n:
        return [chars] if chars else []
    return [chars[i:i + n] for i in range(len(chars) - n + 1)]


def minhash_signature(tokens: list[str], num_perm: int = 128, seed: int = 42) -> list[int]:
    if not tokens:
        return [0] * num_perm
    sig = []
    for i in range(num_perm):
        min_hash = 0xFFFFFFFFFFFFFFFF
        for token in tokens:
            h = hashlib.sha256(f"{seed}:{i}:{token}".encode("utf-8")).digest()
            val = int.from_bytes(h[:8], "big")
            if val < min_hash:
                min_hash = val
        sig.append(min_hash)
    return sig


def minhash_similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b) or len(sig_a) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def minhash_dedup(
    samples: list[dict],
    text_fn,
    threshold: float = 0.7,
    num_perm: int = 128,
    ngram_n: int = 3,
) -> tuple[list[dict], list[int]]:
    import numpy as np

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
