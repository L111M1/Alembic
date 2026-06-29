import logging
from pathlib import Path

import click

from alembic.api.factory import create_client
from alembic.cleaner import DatasetCleaner
from alembic.config import AppConfig, CleanerConfig, ScoringConfig
from alembic.core.inspector import DatasetInspector
from alembic.core.pipeline import Pipeline
from alembic.scoring import DatasetScorer

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
    scfg = pipeline._config.scoring
    print("\n=== Generation Complete ===")
    print(f"  Attempted:  {stats.total_attempted}")
    print(f"  Generated:  {stats.total_generated}")
    print(f"  Filtered:   {stats.total_filtered}")
    if not dry_run:
        out = pipeline._config.output.path
        print(f"  Output:     {out}")
        cleaned = out.replace(".jsonl", "_cleaned.jsonl") if not out.endswith("_cleaned.jsonl") else out
        sc_enabled = scfg.enabled and scfg.dimensions
        if sc_enabled:
            print(f"  Scored:     {scfg.output_path or cleaned.replace('.jsonl', '_scored.jsonl')}")
            if scfg.min_total_score > 0:
                p = scfg.output_path or cleaned.replace('.jsonl', '_scored.jsonl')
                print(f"  Filtered:   {p.replace('_scored.jsonl', '_scored_filtered.jsonl')}")


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
        api_cfg = app_cfg.api
    else:
        scoring_cfg = ScoringConfig()
        api_cfg = None

    if concurrency is not None:
        scoring_cfg.concurrency = concurrency

    api = create_client(
        model=scoring_cfg.model or (api_cfg.model if api_cfg else None),
        api_key=scoring_cfg.api_key or (api_cfg.api_key if api_cfg else None),
        base_url=scoring_cfg.base_url or (api_cfg.base_url if api_cfg else None),
        retry=scoring_cfg.retry or (api_cfg.retry if api_cfg else None),
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
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--samples", "-n", type=int, default=0, help="Number of sample rows to display")
@click.option("--json", "-j", "as_json", is_flag=True, help="Output report as JSON")
@click.option("--quality", "-q", is_flag=True, help="Include MinHash similarity analysis")
@click.option("--graph", "-g", is_flag=True, help="Render Unicode bar charts (topic, length, score distributions)")
def view(input_file: str, samples: int, as_json: bool, quality: bool, graph: bool):
    """View a JSONL dataset: statistics, distribution, and sample rows"""
    inspector = DatasetInspector()
    if as_json:
        import json as json_mod
        report = inspector.inspect_file(input_file)
        if quality:
            report["similarity"] = inspector.analyze_similarity()
        print(json_mod.dumps(report, ensure_ascii=False, indent=2))
    else:
        if graph:
            inspector.inspect_file(input_file)
            if quality:
                inspector.analyze_similarity()
            out = inspector.print_charts(quality=quality)
        else:
            out = inspector.print_report(input_file, quality=quality)
        print(out)
        if samples > 0:
            out2 = inspector.print_samples(input_file, n=samples)
            print(out2)


@main.command()
def list_templates():
    """List available prompt templates"""
    templates_dir = Path(__file__).parent / "prompts" / "templates"
    for f in sorted(templates_dir.glob("*.j2")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
