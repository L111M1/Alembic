import json
from collections import Counter
from pathlib import Path

import numpy as np

from alembic.cleaner.ops import minhash_signature, tokenize_ngrams


def _compute_distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "count": n,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": round(sum(sorted_vals) / n, 1),
        "median": sorted_vals[n // 2] if n % 2 == 1
        else round((sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2, 1),
        "p25": sorted_vals[n // 4],
        "p75": sorted_vals[3 * n // 4],
        "p90": sorted_vals[9 * n // 10] if 9 * n // 10 < n else sorted_vals[-1],
        "p95": sorted_vals[95 * n // 100] if 95 * n // 100 < n else sorted_vals[-1],
    }


def _len_safe(s: str) -> int:
    try:
        return len(s)
    except Exception:
        return 0


class DatasetInspector:
    def __init__(self):
        self._total: int = 0
        self._single_turn: int = 0
        self._multi_turn: int = 0
        self._inst_lengths: list[int] = []
        self._out_lengths: list[int] = []
        self._topics: Counter = Counter()
        self._strategies: Counter = Counter()
        self._scores: Counter = Counter()
        self._score_dims: dict[str, list[float]] = {}
        self._score_dim_names: list[str] = []
        self._sample_texts: list[str] = []

    def inspect_file(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._process(item)

        report = self._build_report()
        return report

    def _process(self, item: dict) -> None:
        self._total += 1

        if "messages" in item and isinstance(item.get("messages"), list) and len(item["messages"]) > 0:
            self._multi_turn += 1
            inst = " ".join(
                m.get("content", "")
                for m in item["messages"]
                if m.get("role") == "user"
            )
            out = " ".join(
                m.get("content", "")
                for m in item["messages"]
                if m.get("role") == "assistant"
            )
        elif "conversations" in item and isinstance(item.get("conversations"), list):
            self._multi_turn += 1
            inst = " ".join(
                t.get("value", "")
                for t in item["conversations"]
                if t.get("from") == "human"
            )
            out = " ".join(
                t.get("value", "")
                for t in item["conversations"]
                if t.get("from") == "gpt"
            )
        else:
            self._single_turn += 1
            inst = item.get("instruction", "")
            out = item.get("output", "") or item.get("response", "")

        self._inst_lengths.append(_len_safe(inst))
        self._out_lengths.append(_len_safe(out))

        if "messages" in item:
            self._sample_texts.append(" ".join(m.get("content", "") for m in item.get("messages", [])))
        else:
            self._sample_texts.append(inst + " " + out)

        meta = item.get("metadata", {})
        if isinstance(meta, dict):
            topic = meta.get("topic")
            if topic:
                self._topics[str(topic)] += 1
            strategy = meta.get("strategy")
            if strategy:
                self._strategies[str(strategy)] += 1

        scores = item.get("scores", {})
        if isinstance(scores, dict) and scores:
            for dim, val in scores.items():
                if dim not in self._score_dims:
                    self._score_dims[dim] = []
                    self._score_dim_names.append(dim)
                try:
                    self._score_dims[dim].append(float(val))
                except (ValueError, TypeError):
                    pass

    def _build_report(self) -> dict:
        report: dict = {
            "total": self._total,
            "format": {
                "single_turn": self._single_turn,
                "multi_turn": self._multi_turn,
            },
            "length_distribution": {
                "instruction": _compute_distribution(self._inst_lengths),
                "output": _compute_distribution(self._out_lengths),
            },
        }

        if self._topics:
            report["topics"] = dict(self._topics.most_common())
        if self._strategies:
            report["strategies"] = dict(self._strategies.most_common())

        if self._score_dims:
            report["scoring"] = {
                dim: _compute_distribution(vals)
                for dim, vals in self._score_dims.items()
            }

        return report

    def analyze_similarity(self, threshold: float = 0.7) -> dict:
        """Compute MinHash pairwise similarity distribution."""
        texts = self._sample_texts
        n = len(texts)
        if n < 2:
            return {"count": n, "near_duplicates": 0, "threshold": threshold}

        num_perm = 128
        signatures = [minhash_signature(tokenize_ngrams(t), num_perm) for t in texts]
        sign_arr = np.array(signatures, dtype=np.uint64)

        max_sims: list[float] = []
        for i in range(n):
            matches = np.sum(sign_arr[i] == sign_arr[i + 1:], axis=1)
            sims = matches.astype(np.float64) / num_perm
            if len(sims) > 0:
                max_sims.append(float(np.max(sims)))

        near = sum(1 for s in max_sims if s >= threshold)
        return {
            "count": n,
            "near_duplicates": near,
            "duplicate_ratio": round(near / max(n, 1), 3),
            "threshold": threshold,
            "max_similarity_distribution": _compute_distribution(max_sims) if max_sims else {"count": 0},
        }

    def print_report(self, path: str, quality: bool = False) -> str:
        report = self.inspect_file(path)
        lines = []
        lines.append("")
        lines.append(f"  Dataset:   {Path(path).name}")
        lines.append(f"  Total:     {report['total']}")
        lines.append(f"  Single:    {report['format']['single_turn']}")
        lines.append(f"  Multi:     {report['format']['multi_turn']}")

        ld = report.get("length_distribution", {})
        for label, key in [("Instruction", "instruction"), ("Output", "output")]:
            dist = ld.get(key)
            if dist and dist.get("count", 0) > 0:
                lines.append(f"  {label} length:  min={dist['min']}, max={dist['max']}, "
                             f"mean={dist['mean']}, median={dist['median']}, "
                             f"p25={dist['p25']}, p75={dist['p75']}, p90={dist['p90']}")

        if "topics" in report:
            lines.append(f"  Topics ({len(report['topics'])}):")
            for topic, cnt in list(report["topics"].items())[:20]:
                lines.append(f"    {topic}: {cnt}")
            if len(report["topics"]) > 20:
                lines.append(f"    ... and {len(report['topics']) - 20} more")

        if "strategies" in report:
            lines.append("  Strategies:")
            for s, cnt in report["strategies"].items():
                lines.append(f"    {s}: {cnt}")

        if "scoring" in report:
            lines.append("  Scores:")
            for dim, dist in report["scoring"].items():
                lines.append(f"    {dim}: min={dist['min']}, max={dist['max']}, "
                             f"mean={dist['mean']}, median={dist['median']}")

        if quality:
            sim = self.analyze_similarity()
            dist = sim.get("max_similarity_distribution", {})
            if dist.get("count", 0) > 0:
                lines.append("  Similarity (max MinHash):")
                lines.append(f"    near-duplicates: {sim['near_duplicates']} ({sim['duplicate_ratio']:.1%} @ >= {sim['threshold']:.0%})")
                lines.append(f"    min={dist['min']:.3f}, max={dist['max']:.3f}, "
                             f"mean={dist['mean']:.3f}, median={dist['median']:.3f}, "
                             f"p25={dist['p25']:.3f}, p75={dist['p75']:.3f}, p90={dist['p90']:.3f}")
            else:
                lines.append(f"  Similarity: {sim['count']} samples, 0 pairwise (insufficient data)")

        lines.append("")
        return "\n".join(lines)

    def print_samples(self, path: str, n: int = 3) -> str:
        lines = []
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if count >= n:
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
                lines.append(f"\n  --- Sample {count} ---")
                if "messages" in item:
                    for m in item["messages"]:
                        role = m.get("role", "?")
                        content = m.get("content", "")
                        if len(content) > 120:
                            content = content[:120] + "..."
                        lines.append(f"  [{role}] {content}")
                elif "conversations" in item:
                    for t in item["conversations"]:
                        speaker = t.get("from", "?")
                        value = t.get("value", "")
                        if len(value) > 120:
                            value = value[:120] + "..."
                        lines.append(f"  [{speaker}] {value}")
                else:
                    inst = item.get("instruction", "")
                    out = item.get("output", "") or item.get("response", "")
                    if len(inst) > 120:
                        inst = inst[:120] + "..."
                    if len(out) > 120:
                        out = out[:120] + "..."
                    lines.append(f"  instruction: {inst}")
                    lines.append(f"  output:      {out}")
        lines.append("")
        return "\n".join(lines)
