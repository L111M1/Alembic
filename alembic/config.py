from dataclasses import dataclass, field
from typing import Optional, Any


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
    checkpoint: bool = False
    checkpoint_path: Optional[str] = None


@dataclass
class CleanerConfig:
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    instruction_min_len: int = 5
    instruction_max_len: int = 4000
    output_min_len: int = 10
    output_max_len: int = 8000
    max_special_char_ratio: float = 0.3
    max_word_repetition_ratio: float = 0.5
    max_char_repetition_ratio: float = 0.5
    dedup: bool = True
    embedding_dedup: bool = False
    embedding_model: str = "text-embedding-3-small"
    embedding_similarity_threshold: float = 0.85
    embedding_batch_size: int = 20


@dataclass
class AppConfig:
    api: APIConfig = field(default_factory=APIConfig)
    strategies: list[StrategyConfig] = field(default_factory=list)
    quality: QualityConfig = field(default_factory=QualityConfig)
    cleaner: CleanerConfig = field(default_factory=CleanerConfig)
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
            remove_html=cleaner_data.get("remove_html", True),
            remove_urls=cleaner_data.get("remove_urls", True),
            remove_emails=cleaner_data.get("remove_emails", True),
            instruction_min_len=int(cleaner_data.get("instruction_min_len", 5)),
            instruction_max_len=int(cleaner_data.get("instruction_max_len", 4000)),
            output_min_len=int(cleaner_data.get("output_min_len", 10)),
            output_max_len=int(cleaner_data.get("output_max_len", 8000)),
            max_special_char_ratio=float(cleaner_data.get("max_special_char_ratio", 0.3)),
            max_word_repetition_ratio=float(cleaner_data.get("max_word_repetition_ratio", 0.5)),
            max_char_repetition_ratio=float(cleaner_data.get("max_char_repetition_ratio", 0.5)),
            dedup=cleaner_data.get("dedup", True),
            embedding_dedup=cleaner_data.get("embedding_dedup", False),
            embedding_model=cleaner_data.get("embedding_model", "text-embedding-3-small"),
            embedding_similarity_threshold=float(cleaner_data.get("embedding_similarity_threshold", 0.85)),
            embedding_batch_size=int(cleaner_data.get("embedding_batch_size", 20)),
        )

        output_cfg = OutputConfig(
            path=output_data.get("path", "./generated_sft.jsonl"),
            format=output_data.get("format", "alpaca"),
            checkpoint=output_data.get("checkpoint", False),
            checkpoint_path=output_data.get("checkpoint_path"),
        )

        return cls(
            api=api_cfg,
            strategies=strategies,
            quality=quality_cfg,
            cleaner=cleaner_cfg,
            output=output_cfg,
            dry_run=data.get("dry_run", False),
            count=int(data.get("count", 100)),
            random_seed=data.get("random_seed"),
        )
