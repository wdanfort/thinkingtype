"""CLI entrypoint for typo-eval."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from typo_eval.config import TypoEvalConfig, load_config, get_repo_root


def _setup_logger(run_dir: Path | None = None) -> logging.Logger:
    """Set up logger with console and optional file output."""
    logger = logging.getLogger("typo_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if run_dir:
        log_path = run_dir / "logs.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _generate_run_id(config: TypoEvalConfig) -> str:
    """Generate a unique run ID."""
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"


def cmd_generate(args: argparse.Namespace, config: TypoEvalConfig, logger: logging.Logger) -> None:
    """Generate input datasets (sentences and artifacts)."""
    from typo_eval.inputs import generate_sentences, generate_artifacts

    repo_root = get_repo_root()
    data_dir = repo_root / "data" / "inputs"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating input datasets...")

    if config.inputs.sentences.get("enabled", True):
        sentences_path = data_dir / "sentences.csv"
        df = generate_sentences(config, sentences_path, seed=config.seed)
        logger.info(f"Generated {len(df)} sentences: {sentences_path}")

    if config.inputs.artifacts.get("enabled", False):
        artifacts_path = data_dir / "artifacts.csv"
        df = generate_artifacts(config, artifacts_path, seed=config.seed)
        logger.info(f"Generated {len(df)} artifacts: {artifacts_path}")

    logger.info("Input generation complete.")


def cmd_render(args: argparse.Namespace, config: TypoEvalConfig, logger: logging.Logger) -> None:
    """Render typography images for all inputs."""
    import pandas as pd
    from typo_eval.fonts import check_fonts_or_exit
    from typo_eval.render import render_sentences, render_artifacts

    repo_root = get_repo_root()

    # Validate fonts
    check_fonts_or_exit(config, repo_root)

    data_dir = repo_root / "data"
    sentences_path = data_dir / "inputs" / "sentences.csv"
    artifacts_path = data_dir / "inputs" / "artifacts.csv"

    logger.info("Rendering typography images...")

    if config.inputs.sentences.get("enabled", True) and sentences_path.exists():
        sentences_df = pd.read_csv(sentences_path)
        rendered_dir = data_dir / "rendered" / "sentences"
        metadata = render_sentences(config, sentences_df, rendered_dir, repo_root)
        metadata.to_csv(data_dir / "inputs" / "sentences_metadata.csv", index=False)
        logger.info(f"Rendered {len(metadata)} sentence images")

    if config.inputs.artifacts.get("enabled", False) and artifacts_path.exists():
        artifacts_df = pd.read_csv(artifacts_path)
        rendered_dir = data_dir / "rendered" / "artifacts"
        metadata = render_artifacts(config, artifacts_df, rendered_dir, repo_root)
        metadata.to_csv(data_dir / "inputs" / "artifacts_metadata.csv", index=False)
        logger.info(f"Rendered {len(metadata)} artifact images")

    logger.info("Rendering complete.")


def cmd_ocr(args: argparse.Namespace, config: TypoEvalConfig, logger: logging.Logger) -> None:
    """Run OCR on rendered images."""
    import pandas as pd
    from typo_eval.ocr import check_tesseract_available, run_ocr_on_sentences, run_ocr_on_artifacts

    if not config.ocr.enabled:
        logger.info("OCR is disabled in config, skipping.")
        return

    if not check_tesseract_available():
        logger.error("Tesseract OCR is not available. Install it with: scripts/install_tesseract.sh")
        sys.exit(1)

    repo_root = get_repo_root()
    data_dir = repo_root / "data"

    sentences_path = data_dir / "inputs" / "sentences.csv"
    artifacts_path = data_dir / "inputs" / "artifacts.csv"

    logger.info("Running OCR on rendered images...")

    if config.inputs.sentences.get("enabled", True) and sentences_path.exists():
        sentences_df = pd.read_csv(sentences_path)
        rendered_dir = data_dir / "rendered" / "sentences"
        ocr_dir = data_dir / "ocr" / "sentences"
        ocr_df = run_ocr_on_sentences(
            sentences_df,
            rendered_dir,
            ocr_dir,
            lang=config.ocr.tesseract_lang,
        )
        ocr_df.to_csv(data_dir / "inputs" / "sentences_ocr.csv", index=False)
        logger.info(f"OCR completed for {len(ocr_df)} sentences")

    if config.inputs.artifacts.get("enabled", False) and artifacts_path.exists():
        artifacts_df = pd.read_csv(artifacts_path)
        rendered_dir = data_dir / "rendered" / "artifacts"
        ocr_dir = data_dir / "ocr" / "artifacts"
        ocr_df = run_ocr_on_artifacts(
            artifacts_df,
            rendered_dir,
            ocr_dir,
            lang=config.ocr.tesseract_lang,
        )
        ocr_df.to_csv(data_dir / "inputs" / "artifacts_ocr.csv", index=False)
        logger.info(f"OCR completed for {len(ocr_df)} artifacts")

    logger.info("OCR complete.")


def cmd_run(args: argparse.Namespace, config: TypoEvalConfig, logger: logging.Logger) -> None:
    """Run inference on all inputs."""
    import pandas as pd
    from typo_eval.inference import run_inference, jsonl_to_csv
    from typo_eval.reporting import generate_manifest, write_manifest

    repo_root = get_repo_root()
    data_dir = repo_root / "data"
    results_dir = repo_root / "results" / "runs"

    # Generate or use provided run_id (args --run-id takes precedence)
    if hasattr(args, "run_id") and args.run_id:
        run_id = args.run_id
    else:
        run_id = config.run_id or _generate_run_id(config)
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update logger with run directory
    logger = _setup_logger(run_dir)
    logger.info(f"Starting inference run: {run_id}")

    # Load input data
    sentences_df = None
    artifacts_df = None
    ocr_sentences_df = None
    ocr_artifacts_df = None
    sentences_metadata_df = None
    artifacts_metadata_df = None

    if config.inputs.sentences.get("enabled", True):
        sentences_path = data_dir / "inputs" / "sentences.csv"
        if sentences_path.exists():
            sentences_df = pd.read_csv(sentences_path)

        ocr_path = data_dir / "inputs" / "sentences_ocr.csv"
        if ocr_path.exists():
            ocr_sentences_df = pd.read_csv(ocr_path)

        metadata_path = data_dir / "inputs" / "sentences_metadata.csv"
        if metadata_path.exists():
            sentences_metadata_df = pd.read_csv(metadata_path)

    if config.inputs.artifacts.get("enabled", False):
        artifacts_path = data_dir / "inputs" / "artifacts.csv"
        if artifacts_path.exists():
            artifacts_df = pd.read_csv(artifacts_path)

        ocr_path = data_dir / "inputs" / "artifacts_ocr.csv"
        if ocr_path.exists():
            ocr_artifacts_df = pd.read_csv(ocr_path)

        metadata_path = data_dir / "inputs" / "artifacts_metadata.csv"
        if metadata_path.exists():
            artifacts_metadata_df = pd.read_csv(metadata_path)

    # Generate and save manifest
    manifest = generate_manifest(run_id, config, repo_root)
    write_manifest(manifest, data_dir / "manifests" / f"run_{run_id}.json")

    # Run inference
    provider = args.provider if hasattr(args, "provider") and args.provider else "openai"
    dry_run = args.dry_run if hasattr(args, "dry_run") else False
    limit = args.limit if hasattr(args, "limit") else None

    jsonl_path = run_inference(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        sentences_df=sentences_df,
        artifacts_df=artifacts_df,
        ocr_sentences_df=ocr_sentences_df,
        ocr_artifacts_df=ocr_artifacts_df,
        sentences_metadata_df=sentences_metadata_df,
        artifacts_metadata_df=artifacts_metadata_df,
        provider_name=provider,
        dry_run=dry_run,
        limit=limit,
    )

    # Convert JSONL to CSV
    csv_path = run_dir / "raw" / "responses.csv"
    jsonl_to_csv(jsonl_path, csv_path)

    logger.info(f"Inference complete. Results saved to {run_dir}")


def cmd_analyze(args: argparse.Namespace, config: TypoEvalConfig, logger: logging.Logger) -> None:
    """Run analysis on inference results."""
    import pandas as pd
    from typo_eval.analysis import analyze_run

    repo_root = get_repo_root()
    data_dir = repo_root / "data"
    results_dir = repo_root / "results" / "runs"

    run_id = args.run_id if hasattr(args, "run_id") and args.run_id else None
    if not run_id:
        # Find most recent run
        runs = sorted(results_dir.iterdir()) if results_dir.exists() else []
        if not runs:
            logger.error("No runs found. Run inference first.")
            sys.exit(1)
        run_id = runs[-1].name
        logger.info(f"Using most recent run: {run_id}")

    run_dir = results_dir / run_id
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Load sentences for category info
    sentences_df = None
    sentences_path = data_dir / "inputs" / "sentences.csv"
    if sentences_path.exists():
        sentences_df = pd.read_csv(sentences_path)

    logger.info(f"Analyzing run: {run_id}")
    analysis_dir = analyze_run(config, run_id, run_dir, sentences_df)
    logger.info(f"Analysis complete. Results saved to {analysis_dir}")


def cmd_compare(args: argparse.Namespace, config: TypoEvalConfig, logger: logging.Logger) -> None:
    """Run cross-run comparison analysis."""
    import pandas as pd
    from typo_eval.comparison import run_comparison_analysis

    repo_root = get_repo_root()
    data_dir = repo_root / "data"
    results_dir = repo_root / "results" / "runs"
    comparison_dir = repo_root / "results" / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison output directory name
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = comparison_dir / f"comparison_{timestamp}"

    # Parse arguments
    run_ids = args.run_ids if hasattr(args, "run_ids") and args.run_ids else None
    provider_filter = args.provider if hasattr(args, "provider") and args.provider else None
    model_filter = args.model if hasattr(args, "model") and args.model else None

    # Load sentences for category info
    sentences_df = None
    sentences_path = data_dir / "inputs" / "sentences.csv"
    if sentences_path.exists():
        sentences_df = pd.read_csv(sentences_path)

    logger.info("Running cross-run comparison analysis...")
    if run_ids:
        logger.info(f"Filtering to run IDs: {run_ids}")
    if provider_filter:
        logger.info(f"Filtering to provider: {provider_filter}")
    if model_filter:
        logger.info(f"Filtering to model containing: {model_filter}")

    run_comparison_analysis(
        output_dir=output_dir,
        run_ids=run_ids,
        provider_filter=provider_filter,
        model_filter=model_filter,
        sentences_df=sentences_df,
        results_dir=results_dir,
    )

    logger.info(f"Comparison analysis complete. Results saved to {output_dir}")


def cmd_all(args: argparse.Namespace, config: TypoEvalConfig, logger: logging.Logger) -> None:
    """Run full pipeline: generate -> render -> ocr -> run -> analyze."""
    logger.info("Running full pipeline...")

    cmd_generate(args, config, logger)
    cmd_render(args, config, logger)
    cmd_ocr(args, config, logger)
    cmd_run(args, config, logger)
    cmd_analyze(args, config, logger)

    logger.info("Full pipeline complete.")


def main() -> None:
    """Main CLI entrypoint."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="typo-eval",
        description="Typography evaluation harness for AI vision models",
    )
    parser.add_argument("--config", default="configs/v0_default.yaml", help="Config file path")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate input datasets")

    # render command
    render_parser = subparsers.add_parser("render", help="Render typography images")

    # ocr command
    ocr_parser = subparsers.add_parser("ocr", help="Run OCR on rendered images")

    # run command (inference)
    run_parser = subparsers.add_parser("run", help="Run inference on all inputs")
    run_parser.add_argument("--run-id", help="Run ID to resume (generates new if not specified)")
    run_parser.add_argument("--dry-run", action="store_true", help="Print planned calls without executing")
    run_parser.add_argument("--limit", type=int, help="Limit number of inference calls")
    run_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google"],
        default="openai",
        help="Inference provider",
    )
    run_parser.add_argument(
        "--input-type",
        choices=["sentences", "artifacts", "both"],
        default="both",
        help="Input type to process",
    )

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze inference results")
    analyze_parser.add_argument("--run", dest="run_id", help="Run ID to analyze")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple runs across providers/models")
    compare_parser.add_argument("--run-ids", nargs="+", help="Specific run IDs to compare")
    compare_parser.add_argument("--provider", choices=["openai", "anthropic", "google"], help="Filter to specific provider")
    compare_parser.add_argument("--model", help="Filter to models containing this substring")

    # all command
    all_parser = subparsers.add_parser("all", help="Run full pipeline")
    all_parser.add_argument("--dry-run", action="store_true", help="Print planned calls without executing")
    all_parser.add_argument("--limit", type=int, help="Limit number of inference calls")
    all_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google"],
        default="openai",
        help="Inference provider",
    )

    args = parser.parse_args()

    logger = _setup_logger()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        sys.exit(1)

    # Dispatch to command handler
    commands = {
        "generate": cmd_generate,
        "render": cmd_render,
        "ocr": cmd_ocr,
        "run": cmd_run,
        "analyze": cmd_analyze,
        "compare": cmd_compare,
        "all": cmd_all,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args, config, logger)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
