import json
from collections import Counter
from pathlib import Path

import numpy as np

from alembic.cleaner.ops import (
    char_repetition_ratio,
    minhash_signature,
    tokenize_ngrams,
    word_repetition_ratio,
)


def _section(label: str, width: int = 68) -> str:
    mid = f" {label} "
    left = (width - len(mid)) // 2
    return "-" * left + mid + "-" * (width - left - len(mid))


def _hist_bins(values: list[float], n_bins: int = 10) -> list[tuple[str, int]]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [(f"{lo:.1f}", len(values))]
    distinct = len(set(round(v, 6) for v in values))
    n_bins = min(n_bins, distinct)
    n_bins = max(2, int(hi - lo) + 1) if hi - lo < n_bins else n_bins
    bins = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in values:
        for i in range(n_bins - 1, -1, -1):
            if v >= bins[i]:
                counts[i] += 1
                break
    result = []
    fmt = ".2f" if hi - lo < 2 else ".0f"
    for i in range(n_bins):
        label = f"{bins[i]:{fmt}}-{bins[i+1]:{fmt}}"
        result.append((label, counts[i]))
    return [r for r in result if r[1] > 0]


def _render_chart(
    items: list[tuple[str, int | float]],
    width: int = 50,
    title: str = "",
    max_label_width: int = 20,
    unit: str = "",
) -> str:
    if not items:
        return ""

    vals = [v for _, v in items]
    max_val = max(vals) if vals else 1
    if max_val == 0:
        return ""

    label_w = min(max(len(lb) for lb, _ in items), max_label_width)
    lines = []
    if title:
        lines.append(f"  {title}")
        lines.append("  " + "-" * (label_w + width + 12 + len(unit)))
    for label, val in items:
        label = label[:label_w].rjust(label_w)
        bar_len = max(1, int(val / (max_val + 1e-9) * width))
        bar = "|" + "#" * bar_len + " " * (width - bar_len) + "|"
        lines.append(f"  {label} {bar} {val}{unit}")
    return "\n".join(lines)


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
        self._max_sims: list[float] = []
        self._sim_threshold: float = 0.7
        self._similar_pairs: list[tuple[int, int, float]] = []
        self._word_rep_ratios: list[float] = []
        self._char_rep_ratios: list[float] = []
        self._empty_inst: int = 0
        self._empty_out: int = 0
        self._parse_errors: int = 0

    def inspect_file(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    self._parse_errors += 1
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

        text = self._sample_texts[-1]
        self._word_rep_ratios.append(word_repetition_ratio(text))
        self._char_rep_ratios.append(char_repetition_ratio(text))
        if not inst.strip():
            self._empty_inst += 1
        if not out.strip():
            self._empty_out += 1

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
        self._sim_threshold = threshold
        texts = self._sample_texts
        n = len(texts)
        if n < 2:
            self._max_sims = []
            return {"count": n, "near_duplicates": 0, "threshold": threshold}

        num_perm = 128
        signatures = [minhash_signature(tokenize_ngrams(t), num_perm) for t in texts]
        sign_arr = np.array(signatures, dtype=np.uint64)

        self._max_sims = []
        self._similar_pairs = []
        for i in range(n):
            matches = np.sum(sign_arr[i] == sign_arr[i + 1:], axis=1)
            sims = matches.astype(np.float64) / num_perm
            for j_offset, sim in enumerate(sims):
                s = float(sim)
                self._similar_pairs.append((i, i + 1 + j_offset, s))
            if len(sims) > 0:
                self._max_sims.append(float(np.max(sims)))

        self._similar_pairs.sort(key=lambda x: -x[2])

        near = sum(1 for s in self._max_sims if s >= threshold)
        return {
            "count": n,
            "near_duplicates": near,
            "duplicate_ratio": round(near / max(n, 1), 3),
            "threshold": threshold,
            "max_similarity_distribution": _compute_distribution(self._max_sims) if self._max_sims else {"count": 0},
        }

    def print_charts(self, quality: bool = False) -> str:
        W = 40
        S = "=" * 68

        out: list[str] = []
        out.append(S)
        out.append("  DATA PROFILE")
        out.append(f"  {self._total} samples  |  "
                   f"{self._single_turn} single-turn  |  "
                   f"{self._multi_turn} multi-turn")

        if self._total > 0 and self._multi_turn > 0:
            out.append("")
            out.append(_section("COMPOSITION"))
            out.append(_render_chart(
                [("1-turn", self._single_turn), ("multi", self._multi_turn)],
                width=W, title="Format",
            ))

        if self._strategies and len(self._strategies) > 1:
            out.append("")
            out.append(_section("STRATEGIES"))
            items = [(s[:14], c) for s, c in self._strategies.most_common()]
            out.append(_render_chart(items, width=W, title="Generation Strategy"))

        if self._topics and len(self._topics) > 1:
            out.append("")
            out.append(_section("TOPICS"))
            items = [(t[:14], c) for t, c in self._topics.most_common(15)]
            out.append(_render_chart(items, width=W, title="Topic Coverage"))

        if self._inst_lengths or self._out_lengths:
            out.append("")
            out.append(_section("LENGTH"))
        if self._inst_lengths:
            inst = self._inst_lengths
            out.append(f"  instruction  |  "
                       f"min={min(inst)} max={max(inst)} "
                       f"mean={sum(inst)//len(inst):.0f} "
                       f"p50={sorted(inst)[len(inst)//2]}")
            out.append(_render_chart(
                _hist_bins([float(x) for x in inst], n_bins=8),
                width=W, title="Instruction Histogram",
            ))
        if self._out_lengths:
            outl = self._out_lengths
            out.append(f"  output        |  "
                       f"min={min(outl)} max={max(outl)} "
                       f"mean={sum(outl)//len(outl):.0f} "
                       f"p50={sorted(outl)[len(outl)//2]}")
            out.append(_render_chart(
                _hist_bins([float(x) for x in outl], n_bins=8),
                width=W, title="Output Histogram",
            ))

        if self._score_dims:
            out.append("")
            out.append(_section("SCORES"))
            for dim, vals in self._score_dims.items():
                out.append(f"  {dim:14}  |  "
                           f"min={min(vals):.0f} max={max(vals):.0f} "
                           f"mean={sum(vals)/len(vals):.1f} "
                           f"p50={sorted(vals)[len(vals)//2]:.0f}")
                out.append(_render_chart(
                    _hist_bins(vals, n_bins=8),
                    width=W, title=f"Score Histogram [{dim}]",
                ))

        if quality:
            out.append("")
            out.append(_section("QUALITY"))
            issues: list[str] = []
            if self._parse_errors:
                issues.append(f"{self._parse_errors} parse errors")
            if self._empty_inst:
                issues.append(f"{self._empty_inst} empty instructions")
            if self._empty_out:
                issues.append(f"{self._empty_out} empty outputs")
            out.append(f"  issues:  {', '.join(issues) if issues else 'none'}")
            if self._word_rep_ratios:
                w = self._word_rep_ratios
                out.append(f"  word rept  |  "
                           f"max={max(w):.2f} mean={sum(w)/len(w):.3f} "
                           f">0.1: {sum(1 for r in w if r > 0.1)}")
                out.append(_render_chart(
                    _hist_bins(w, n_bins=6),
                    width=W, title="Word Repetition",
                ))
            if self._char_rep_ratios:
                c = self._char_rep_ratios
                out.append(f"  char rept  |  "
                           f"max={max(c):.3f} mean={sum(c)/len(c):.4f} "
                           f">0.02: {sum(1 for r in c if r > 0.02)}")
                out.append(_render_chart(
                    _hist_bins(c, n_bins=6),
                    width=W, title="Char Repetition",
                ))

            out.append("")
            out.append(_section("SIMILARITY (MinHash)"))
            if self._max_sims:
                near = sum(1 for s in self._max_sims if s >= self._sim_threshold)
                out.append(f"  threshold {self._sim_threshold:.0%}  |  "
                           f"{near}/{len(self._max_sims)} near-duplicates "
                           f"({near / max(len(self._max_sims), 1):.1%})")
                out.append(_render_chart(
                    _hist_bins(self._max_sims, n_bins=8),
                    width=W, title="Max Pairwise Similarity",
                ))
                if self._similar_pairs:
                    top = self._similar_pairs[:8]
                    items = [(f"#{i}-#{j}", round(s, 3)) for i, j, s in top]
                    out.append(_render_chart(
                        items, width=W, title="Top Similar Pairs",
                    ))
            else:
                out.append("  (need >= 2 samples)")

        out.append(S)
        return "\n".join(out)

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
        lines.append("")
        lines.append(_section("SAMPLES"))
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
                lines.append(f"\n  [Sample {count}]")
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
