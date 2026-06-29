import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from alembic.api.base import BaseAPIClient, RetryConfig, retry_with_backoff
from alembic.config import ScoringConfig
from alembic.prompts.builder import PromptBuilder

logger = logging.getLogger(__name__)

_SCORING_TEMPLATE_SYSTEM = "scorer_system.j2"
_SCORING_TEMPLATE_USER = "scorer_user.j2"


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
        if not samples:
            logger.warning("No valid samples to score")
            return 0, 0

        concurrency = max(1, self._config.concurrency)

        if concurrency <= 1:
            scored = self._score_sequential(api, samples)
        else:
            scored = self._score_parallel(api, samples, concurrency)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in scored:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Scoring done: scored={self._scored_count}, failed={self._failed_count}")
        return self._scored_count, self._failed_count

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
            scored = self._score_one(api, sample, i)
            if scored:
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
        try:
            return retry_with_backoff(
                lambda: self._score_one(api, sample, index),
                RetryConfig(max_retries=2),
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

        dimensions = self._config.dimensions
        dim_desc = "\n".join(
            f"- {d['name']} ({d.get('label', d['name'])}): {d.get('description', '')}. 分值范围 1-{d.get('max_score', 10)}"
            for d in dimensions
        )
        dim_names = ", ".join(f'"{d["name"]}"' for d in dimensions)

        prompt = PromptBuilder(lang=self._config.lang)
        prompt.from_template(_SCORING_TEMPLATE_SYSTEM, dimensions=dim_desc)
        prompt.from_template(
            _SCORING_TEMPLATE_USER,
            instruction=inst,
            output=out,
            dim_names=dim_names,
        )
        messages = prompt.build()

        temp = self._config.params.get("temperature", 0.3)
        max_tok = self._config.params.get("max_tokens", 1024)

        raw = api.call(messages, temperature=temp, max_tokens=max_tok)
        scores = self._parse_scores(raw, dimensions)

        result = dict(sample)
        result["scores"] = scores
        result["total_score"] = sum(scores.values()) if scores else 0
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
