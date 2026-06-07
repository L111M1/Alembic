import click
import logging
from pathlib import Path

from alembic.core.pipeline import Pipeline
from alembic.cleaner import DatasetCleaner
from alembic.scoring import DatasetScorer
from alembic.config import CleanerConfig, AppConfig, ScoringConfig
from alembic.api.factory import create_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@click.group()
def main():
    """Alembic - distill raw data into SFT training gold"""


@main.command()
@click.option("--config", "-c", required=True, help="YAML config file path")
@click.option("--dry-run", is_flag=True, help="Preview mode, do not write files")
@click.option("--count", "-n", type=int, default=None, help="Override generation count")
@click.option("--seed", type=int, default=None, help="Random seed")
def generate(config: str, dry_run: bool, count: int, seed: int):
    """Generate SFT training data"""
    pipeline = Pipeline.from_yaml(config)
    if dry_run:
        pipeline._config.dry_run = True
    if count is not None:
        pipeline._config.count = count
    if seed is not None:
        pipeline._config.random_seed = seed

    stats = pipeline.run()
    print("\n=== Generation Complete ===")
    print(f"  Attempted:  {stats.total_attempted}")
    print(f"  Generated:  {stats.total_generated}")
    print(f"  Filtered:   {stats.total_filtered}")
    if not dry_run:
        print(f"  Output:     {pipeline._config.output.path}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output file path (default: input_cleaned.jsonl)")
@click.option("--config", "-c", default=None, help="YAML config with cleaner section")
def clean(input_file: str, output: str, config: str):
    """Clean a JSONL dataset"""
    if config:
        app_cfg = AppConfig.from_yaml(config)
        cleaner_cfg = app_cfg.cleaner
    else:
        cleaner_cfg = CleanerConfig()

    if output is None:
        p = Path(input_file)
        output = str(p.parent / f"{p.stem}_cleaned.jsonl")

    cleaner = DatasetCleaner(cleaner_cfg)
    kept, dropped = cleaner.clean_file(input_file, output)
    print("\n=== Clean Complete ===")
    print(f"  Kept:    {kept}")
    print(f"  Dropped: {dropped}")
    print(f"  Output:  {output}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output file path (default: input_scored.jsonl)")
@click.option("--config", "-c", default=None, help="YAML config with scoring section")
@click.option("--concurrency", "-n", type=int, default=None, help="Override concurrency")
def score(input_file: str, output: str, config: str, concurrency: int):
    """Score a JSONL dataset using an LLM judge"""
    if config:
        app_cfg = AppConfig.from_yaml(config)
        scoring_cfg = app_cfg.scoring
    else:
        scoring_cfg = ScoringConfig()

    if concurrency is not None:
        scoring_cfg.concurrency = concurrency

    api = create_client(
        model=scoring_cfg.model,
        api_key=scoring_cfg.api_key,
        base_url=scoring_cfg.base_url,
        retry=scoring_cfg.retry,
    )

    if output is None:
        p = Path(input_file)
        output = str(p.parent / f"{p.stem}_scored.jsonl")

    scorer = DatasetScorer(scoring_cfg)
    scored, failed = scorer.score_file(api, input_file, output)
    print("\n=== Scoring Complete ===")
    print(f"  Scored:  {scored}")
    print(f"  Failed:  {failed}")
    print(f"  Output:  {output}")


@main.command()
def list_templates():
    """List available prompt templates"""
    templates_dir = Path(__file__).parent / "prompts" / "templates"
    for f in sorted(templates_dir.glob("*.j2")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
