from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIConfig:
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    lang: str = "en"
    concurrency: int = 1
    params: dict = field(default_factory=dict)
    retry: dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    type: str = ""
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class QualityConfig:
    instruction_min_len: int = 5
    instruction_max_len: int = 4000
    output_min_len: int = 10
    output_max_len: int = 8000
    dedup: bool = True
    remove_truncated: bool = True


@dataclass
class OutputConfig:
    path: str = "./generated_sft.jsonl"
    format: str = "alpaca"
    multi_turn: bool = False
    checkpoint: bool = False
    checkpoint_path: Optional[str] = None


@dataclass
class CleanerConfig:
    instruction_min_len: int = 5
    instruction_max_len: int = 4000
    output_min_len: int = 10
    output_max_len: int = 8000
    max_special_char_ratio: float = 0.3
    max_word_repetition_ratio: float = 0.5
    max_char_repetition_ratio: float = 0.5
    min_ngram_diversity: float = 0.2
    ngram_diversity_n: int = 3
    ngram_diversity_unit: str = "char"
    minhash_dedup: bool = True
    minhash_threshold: float = 0.7
    minhash_num_perm: int = 128
    minhash_ngram_n: int = 3
    embedding_dedup: bool = False
    embedding_model: str = "text-embedding-3-small"
    embedding_similarity_threshold: float = 0.85
    embedding_batch_size: int = 20
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None
    input_format: str = "alpaca"  # "alpaca" or "chatml"
    field_map: Optional[dict] = None  # e.g. {"instruction": "question", "output": "answer"}


@dataclass
class ScoringConfig:
    enabled: bool = False
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    lang: str = "en"
    concurrency: int = 3
    dimensions: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    retry: dict = field(default_factory=dict)
    min_total_score: float = 0.0
    output_path: Optional[str] = None
    field_map: Optional[dict] = None
    judges: list = field(default_factory=list)
    aggregation: str = "mean"
    min_judges: int = 1
    max_judge_disagreement: float = 0.0


@dataclass
class AppConfig:
    api: APIConfig = field(default_factory=APIConfig)
    strategies: list[StrategyConfig] = field(default_factory=list)
    quality: QualityConfig = field(default_factory=QualityConfig)
    cleaner: CleanerConfig = field(default_factory=CleanerConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    dry_run: bool = False
    count: int = 100
    random_seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        api_data = data.get("api", {})
        strategies_data = data.get("strategies", [])
        quality_data = data.get("quality", {})
        cleaner_data = data.get("cleaner", {})
        scoring_data = data.get("scoring", {})
        output_data = data.get("output", {})

        api_cfg = APIConfig(
            model=api_data.get("model", "gpt-4o"),
            api_key=api_data.get("api_key"),
            base_url=api_data.get("base_url"),
            lang=api_data.get("lang", "en"),
            concurrency=int(api_data.get("concurrency", 1)),
            params=api_data.get("params", {}),
            retry=api_data.get("retry", {}),
        )

        strategies = [
            StrategyConfig(
                type=s.get("type", ""),
                weight=float(s.get("weight", 1.0)),
                params={k: v for k, v in s.items() if k not in ("type", "weight")},
            )
            for s in strategies_data
        ]

        quality_cfg = QualityConfig(
            instruction_min_len=int(quality_data.get("instruction_min_len", 5)),
            instruction_max_len=int(quality_data.get("instruction_max_len", 4000)),
            output_min_len=int(quality_data.get("output_min_len", 10)),
            output_max_len=int(quality_data.get("output_max_len", 8000)),
            dedup=quality_data.get("dedup", True),
            remove_truncated=quality_data.get("remove_truncated", True),
        )

        cleaner_cfg = CleanerConfig(
            instruction_min_len=int(cleaner_data.get("instruction_min_len", 5)),
            instruction_max_len=int(cleaner_data.get("instruction_max_len", 4000)),
            output_min_len=int(cleaner_data.get("output_min_len", 10)),
            output_max_len=int(cleaner_data.get("output_max_len", 8000)),
            max_special_char_ratio=float(cleaner_data.get("max_special_char_ratio", 0.3)),
            max_word_repetition_ratio=float(cleaner_data.get("max_word_repetition_ratio", 0.5)),
            max_char_repetition_ratio=float(cleaner_data.get("max_char_repetition_ratio", 0.5)),
            min_ngram_diversity=float(cleaner_data.get("min_ngram_diversity", 0.2)),
            ngram_diversity_n=int(cleaner_data.get("ngram_diversity_n", 3)),
            ngram_diversity_unit=str(cleaner_data.get("ngram_diversity_unit", "char")),
            minhash_dedup=cleaner_data.get("minhash_dedup", True),
            minhash_threshold=float(cleaner_data.get("minhash_threshold", 0.7)),
            minhash_num_perm=int(cleaner_data.get("minhash_num_perm", 128)),
            minhash_ngram_n=int(cleaner_data.get("minhash_ngram_n", 3)),
            embedding_dedup=cleaner_data.get("embedding_dedup", False),
            embedding_model=cleaner_data.get("embedding_model", "text-embedding-3-small"),
            embedding_similarity_threshold=float(cleaner_data.get("embedding_similarity_threshold", 0.85)),
            embedding_batch_size=int(cleaner_data.get("embedding_batch_size", 20)),
            embedding_api_key=cleaner_data.get("embedding_api_key"),
            embedding_base_url=cleaner_data.get("embedding_base_url"),
            input_format=cleaner_data.get("input_format", "alpaca"),
            field_map=cleaner_data.get("field_map"),
        )

        scoring_cfg = ScoringConfig(
            enabled=scoring_data.get("enabled", False),
            model=scoring_data.get("model"),
            api_key=scoring_data.get("api_key"),
            base_url=scoring_data.get("base_url"),
            lang=scoring_data.get("lang", "en"),
            concurrency=int(scoring_data.get("concurrency", 3)),
            dimensions=scoring_data.get("dimensions", []),
            params=scoring_data.get("params", {}),
            retry=scoring_data.get("retry", {}),
            min_total_score=float(scoring_data.get("min_total_score", 0.0)),
            output_path=scoring_data.get("output_path"),
            field_map=scoring_data.get("field_map"),
            judges=scoring_data.get("judges", []),
            aggregation=scoring_data.get("aggregation", "mean"),
            min_judges=max(1, int(scoring_data.get("min_judges", 1))),
            max_judge_disagreement=float(
                scoring_data.get("max_judge_disagreement", 0.0)
            ),
        )

        output_cfg = OutputConfig(
            path=output_data.get("path", "./generated_sft.jsonl"),
            format=output_data.get("format", "alpaca"),
            multi_turn=output_data.get("multi_turn", False),
            checkpoint=output_data.get("checkpoint", False),
            checkpoint_path=output_data.get("checkpoint_path"),
        )

        return cls(
            api=api_cfg,
            strategies=strategies,
            quality=quality_cfg,
            cleaner=cleaner_cfg,
            scoring=scoring_cfg,
            output=output_cfg,
            dry_run=data.get("dry_run", False),
            count=int(data.get("count", 0)),
            random_seed=data.get("random_seed"),
        )
