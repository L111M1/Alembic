import json
import logging
import time
from collections import Counter
from pathlib import Path

from alembic.core.observer import Observer

logger = logging.getLogger(__name__)


class StatisticsCollector(Observer):
    def __init__(self):
        self._start_time: float = 0
        self._end_time: float = 0
        self._total: int = 0
        self._generated: int = 0
        self._filtered: int = 0
        self._by_strategy: Counter = Counter()
        self._by_topic: Counter = Counter()
        self._inst_lengths: list[int] = []
        self._out_lengths: list[int] = []
        self._errors: list[dict] = []
        self._cleaner_kept: int = 0
        self._cleaner_dropped: int = 0
        self._scorer_scored: int = 0
        self._scorer_failed: int = 0
        self._score_distributions: dict[str, list[float]] = {}
        self._total_scores: list[float] = []
        self._score_filter_kept: int = 0
        self._score_filter_dropped: int = 0

    def on_start(self, total: int) -> None:
        self._start_time = time.time()
        self._total = total

    def on_sample(self, index: int, success: bool, strategy: str) -> None:
        pass

    def on_error(self, index: int, strategy: str, error: str) -> None:
        pass

    def on_complete(self, stats) -> None:
        self._end_time = time.time()
        self._generated = stats.total_generated
        self._filtered = stats.total_filtered
        self._by_strategy = Counter(stats.by_strategy or {})
        self._errors = stats.errors or []

    def record_sample(self, sample: dict) -> None:
        if "messages" in sample and isinstance(sample["messages"], list):
            inst = " ".join(m["content"] for m in sample["messages"] if m.get("role") == "user")
            out = " ".join(m["content"] for m in sample["messages"] if m.get("role") == "assistant")
        else:
            inst = sample.get("instruction", "")
            out = sample.get("output", "") or sample.get("response", "")
        self._inst_lengths.append(len(inst))
        self._out_lengths.append(len(out))
        meta = sample.get("metadata", {})
        if meta.get("topic"):
            self._by_topic[meta["topic"]] += 1

    def record_cleaner(self, kept: int, dropped: int) -> None:
        self._cleaner_kept = kept
        self._cleaner_dropped = dropped

    def record_scorer(self, scored: int, failed: int) -> None:
        self._scorer_scored = scored
        self._scorer_failed = failed

    def record_scores(self, scored_path: str) -> None:
        with open(scored_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scores = item.get("scores", {})
                for dim, val in scores.items():
                    if dim not in self._score_distributions:
                        self._score_distributions[dim] = []
                    self._score_distributions[dim].append(float(val))
                ts = item.get("total_score")
                if ts is not None:
                    self._total_scores.append(float(ts))

    def record_score_filter(self, kept: int, dropped: int) -> None:
        self._score_filter_kept = kept
        self._score_filter_dropped = dropped

    def generate_report(self) -> dict:
        elapsed = self._end_time - self._start_time if self._end_time > 0 else 0
        report = {
            "pipeline": {
                "total_attempted": self._total,
                "total_generated": self._generated,
                "total_filtered": self._filtered,
                "pass_rate": round(self._generated / max(self._total, 1), 3),
                "elapsed_seconds": round(elapsed, 1),
                "generation_rate": round(self._generated / max(elapsed, 0.001), 2),
            },
            "by_strategy": dict(Counter(self._by_strategy).most_common()),
            "by_topic": dict(Counter(self._by_topic).most_common()) if self._by_topic else None,
            "length_distribution": self._compute_length_stats(),
            "cleaner": {
                "kept": self._cleaner_kept,
                "dropped": self._cleaner_dropped,
                "retention_rate": round(self._cleaner_kept / max(self._cleaner_kept + self._cleaner_dropped, 1), 3),
            },
        }

        if self._scorer_scored > 0 or self._scorer_failed > 0:
            report["scorer"] = {
                "scored": self._scorer_scored,
                "failed": self._scorer_failed,
                "success_rate": round(self._scorer_scored / max(self._scorer_scored + self._scorer_failed, 1), 3),
            }
            if self._total_scores:
                report["scorer"]["total_score_distribution"] = self._compute_distribution(self._total_scores)
            if self._score_distributions:
                report["scorer"]["dimension_distributions"] = {
                    dim: self._compute_distribution(vals)
                    for dim, vals in self._score_distributions.items()
                }
            if self._score_filter_kept > 0 or self._score_filter_dropped > 0:
                report["scorer"]["score_filter"] = {
                    "kept": self._score_filter_kept,
                    "dropped": self._score_filter_dropped,
                    "retention_rate": round(self._score_filter_kept / max(self._score_filter_kept + self._score_filter_dropped, 1), 3),
                }

        return report

    def _compute_length_stats(self) -> dict:
        inst = self._inst_lengths
        out = self._out_lengths
        return {
            "instruction_length": self._compute_distribution(inst) if inst else None,
            "output_length": self._compute_distribution(out) if out else None,
        }

    @staticmethod
    def _compute_distribution(values: list) -> dict:
        if not values:
            return {"count": 0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "count": n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mean": round(sum(sorted_vals) / n, 1),
            "median": sorted_vals[n // 2] if n % 2 == 1 else round((sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2, 1),
            "p25": sorted_vals[n // 4],
            "p75": sorted_vals[3 * n // 4],
            "p90": sorted_vals[9 * n // 10],
            "p95": sorted_vals[95 * n // 100],
        }

    def save_report(self, path: str) -> str:
        report = self.generate_report()
        p = Path(path)
        out = str(p.parent / f"{p.stem}_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Stats report saved to {out}")
        return out
