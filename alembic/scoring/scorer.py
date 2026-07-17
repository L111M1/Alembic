import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from alembic.api.base import BaseAPIClient, RetryConfig, retry_with_backoff
from alembic.config import ScoringConfig
from alembic.prompts.builder import PromptBuilder

logger = logging.getLogger(__name__)

_SCORING_TEMPLATE_SYSTEM = "scorer_system.j2"
_SCORING_TEMPLATE_USER = "scorer_user.j2"


def _default_rubric(max_score: int, lang: str = "en") -> list[dict]:
    step = max_score // 4
    descriptions = (
        [
            "较差——错误、误导或无关",
            "一般——大体正确但存在明显缺漏",
            "良好——正确、实用并覆盖主要要点",
            "优秀——准确、完整且符合最佳实践",
        ]
        if lang == "zh"
        else [
            "Poor — incorrect, misleading, or irrelevant",
            "Fair — mostly correct but has notable gaps",
            "Good — correct, useful, covers main points",
            "Excellent — precise, thorough, best-practice",
        ]
    )
    return [
        {"range": f"1-{step}", "desc": descriptions[0]},
        {"range": f"{step+1}-{2*step}", "desc": descriptions[1]},
        {"range": f"{2*step+1}-{3*step}", "desc": descriptions[2]},
        {"range": f"{3*step+1}-{max_score}", "desc": descriptions[3]},
    ]


class DatasetScorer:
    def __init__(self, config: ScoringConfig):
        self._config = config
        self._scored_count = 0
        self._failed_count = 0

    def score_file(
        self,
        api: BaseAPIClient,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> tuple[int, int]:
        if output_path is None:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_scored.jsonl")

        samples = self._load_samples(input_path)
        scored = self.score_samples(api, samples)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in scored:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Scoring done: scored={self._scored_count}, failed={self._failed_count}")
        return self._scored_count, self._failed_count

    def score_samples(
        self, api: BaseAPIClient, samples: list[dict],
    ) -> list[dict]:
        if not samples:
            logger.warning("No valid samples to score")
            return []
        concurrency = max(1, self._config.concurrency)
        if concurrency <= 1:
            return self._score_sequential(api, samples)
        return self._score_parallel(api, samples, concurrency)

    def _load_samples(self, input_path: str) -> list[dict]:
        samples = []
        field_map = self._config.field_map
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if field_map:
                    sample = {v: sample.get(k, "") for k, v in field_map.items()}
                inst = sample.get("instruction", "")
                out = sample.get("output", "") or sample.get("response", "")
                msgs = sample.get("messages")
                if (inst and out) or (msgs and isinstance(msgs, list) and len(msgs) >= 2):
                    samples.append(sample)
        return samples

    def _score_sequential(self, api: BaseAPIClient, samples: list[dict]) -> list[dict]:
        results = []
        for i, sample in enumerate(samples):
            scored = self._score_one_safe(api, sample, i)
            if scored and "scores" in scored:
                results.append(scored)
                self._scored_count += 1
            else:
                results.append(sample)
                self._failed_count += 1
        return results

    def _score_parallel(self, api: BaseAPIClient, samples: list[dict], concurrency: int) -> list[dict]:
        results = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(self._score_one_safe, api, sample, i): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result and "scores" in result:
                        self._scored_count += 1
                    else:
                        self._failed_count += 1
                    results[idx] = result if result else samples[idx]
                except Exception as e:
                    logger.warning(f"Scoring failed for sample {idx}: {e}")
                    self._failed_count += 1
                    results[idx] = samples[idx]
        return results

    def _score_one_safe(self, api: BaseAPIClient, sample: dict, index: int) -> Optional[dict]:
        rt = self._config.retry
        rc = RetryConfig(
            max_retries=rt.get("max_retries", 3),
            initial_delay=rt.get("initial_delay", 1.0),
            backoff_multiplier=rt.get("backoff_multiplier", 2.0),
            max_delay=rt.get("max_delay", 30.0),
        )
        try:
            return retry_with_backoff(
                lambda: self._score_one(api, sample, index),
                rc,
                f"Score sample {index}",
            )
        except RuntimeError as e:
            logger.warning(str(e))
            return None

    def _score_one(self, api: BaseAPIClient, sample: dict, index: int) -> Optional[dict]:
        if "messages" in sample and isinstance(sample["messages"], list):
            inst = "\n".join(f"[{m['role']}]: {m['content']}" for m in sample["messages"])
            out = inst
        else:
            inst = sample.get("instruction", "")
            out = sample.get("output", "") or sample.get("response", "")

        dims = [{**d} for d in self._config.dimensions]
        for d in dims:
            if "label" not in d:
                d["label"] = d["name"]
            if "rubric" not in d:
                d["rubric"] = _default_rubric(
                    d.get("max_score", 10), self._config.lang
                )
        dim_names = ", ".join(f'"{d["name"]}"' for d in dims)

        prompt = PromptBuilder(lang=self._config.lang)
        prompt.from_template(_SCORING_TEMPLATE_SYSTEM, dimensions=dims)
        prompt.from_template(
            _SCORING_TEMPLATE_USER,
            instruction=inst,
            output=out,
            source_text=(sample.get("metadata") or {}).get("source_text", ""),
            dim_names=dim_names,
        )
        messages = prompt.build()

        temp = self._config.params.get("temperature", 0.3)
        max_tok = self._config.params.get("max_tokens", 1024)

        raw = api.call(messages, temperature=temp, max_tokens=max_tok)
        scores = self._parse_scores(raw, dims)

        if not scores:
            raise ValueError(f"Failed to parse judge response as JSON: {raw[:200]}")

        result = dict(sample)
        result["scores"] = scores
        result["total_score"] = sum(scores.values())
        return result

    def _parse_scores(self, raw: str, dimensions: list[dict]) -> dict:
        from alembic.api.base import _extract_json

        try:
            data = _extract_json(raw)
        except ValueError:
            logger.warning(f"Failed to parse judge response as JSON: {raw[:200]}")
            return {}

        scores = {}
        for dim in dimensions:
            name = dim["name"]
            if name in data:
                try:
                    scores[name] = float(data[name])
                except (ValueError, TypeError):
                    scores[name] = 0
        return scores

    @property
    def stats(self) -> tuple[int, int]:
        return self._scored_count, self._failed_count


class MultiJudgeScorer:
    """Score each sample with multiple independent judges and aggregate results."""

    def __init__(self, config: ScoringConfig):
        if config.aggregation not in {"mean", "min", "max", "median"}:
            raise ValueError("scoring.aggregation must be mean, min, max, or median")
        self._config = config
        self._scored_count = 0
        self._failed_count = 0

    def score_samples(
        self,
        judges: list[tuple[str, BaseAPIClient]],
        samples: list[dict],
    ) -> list[dict]:
        if not samples or not judges:
            return samples
        judge_results: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=len(judges)) as executor:
            futures = {
                executor.submit(
                    DatasetScorer(self._config).score_samples,
                    api,
                    [dict(sample) for sample in samples],
                ): name
                for name, api in judges
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    judge_results[name] = future.result()
                except Exception as exc:
                    logger.warning("Judge %s failed: %s", name, exc)

        results: list[dict] = []
        for index, sample in enumerate(samples):
            result = dict(sample)
            per_judge: dict[str, dict] = {}
            for name, rows in judge_results.items():
                if index >= len(rows):
                    continue
                scores = rows[index].get("scores")
                if isinstance(scores, dict) and scores:
                    per_judge[name] = scores
            result["judge_scores"] = per_judge
            result["judge_count"] = len(per_judge)
            if len(per_judge) < self._config.min_judges:
                result["cross_validation_failed"] = True
                result["scores"] = {}
                result["total_score"] = 0.0
                result["judge_disagreement"] = 0.0
                self._failed_count += 1
                results.append(result)
                continue

            aggregated: dict[str, float] = {}
            disagreements: list[float] = []
            for dimension in self._config.dimensions:
                name = dimension["name"]
                values = [
                    float(scores[name])
                    for scores in per_judge.values()
                    if name in scores
                ]
                if not values:
                    continue
                aggregated[name] = self._aggregate(values)
                disagreements.append(max(values) - min(values))
            result["scores"] = aggregated
            result["total_score"] = sum(aggregated.values())
            result["judge_disagreement"] = max(disagreements, default=0.0)
            result["cross_validation_failed"] = False
            self._scored_count += 1
            results.append(result)
        return results

    def _aggregate(self, values: list[float]) -> float:
        if self._config.aggregation == "min":
            return min(values)
        if self._config.aggregation == "max":
            return max(values)
        if self._config.aggregation == "median":
            return float(statistics.median(values))
        return sum(values) / len(values)

    @property
    def stats(self) -> tuple[int, int]:
        return self._scored_count, self._failed_count
