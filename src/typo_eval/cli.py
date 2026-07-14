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
    text_sentences_df = None
    text_artifacts_df = None
    sentences_metadata_df = None
    artifacts_metadata_df = None

    if config.inputs.sentences.get("enabled", True):
        sentences_path = data_dir / "inputs" / "sentences.csv"
        if sentences_path.exists():
            sentences_df = pd.read_csv(sentences_path)
            # Use sentences directly as text input (no OCR needed)
            text_sentences_df = sentences_df.copy()

        metadata_path = data_dir / "inputs" / "sentences_metadata.csv"
        if metadata_path.exists():
            sentences_metadata_df = pd.read_csv(metadata_path)

    if config.inputs.artifacts.get("enabled", False):
        artifacts_path = data_dir / "inputs" / "artifacts.csv"
        if artifacts_path.exists():
            artifacts_df = pd.read_csv(artifacts_path)
            # Use artifacts directly as text input (no OCR needed)
            text_artifacts_df = artifacts_df.copy()

        metadata_path = data_dir / "inputs" / "artifacts_metadata.csv"
        if metadata_path.exists():
            artifacts_metadata_df = pd.read_csv(metadata_path)

    # Optional sharding for parallel workers (disjoint sentence_id sets)
    shard = getattr(args, "shard", None)
    if shard:
        shard_i, shard_n = map(int, shard.split("/"))
        if sentences_df is not None:
            mask = sentences_df["sentence_id"] % shard_n == shard_i
            sentences_df = sentences_df[mask]
            text_sentences_df = text_sentences_df[mask]
        if sentences_metadata_df is not None:
            sentences_metadata_df = sentences_metadata_df[
                sentences_metadata_df["sentence_id"] % shard_n == shard_i
            ]
        logger.info(f"Shard {shard_i}/{shard_n}: {0 if sentences_df is None else len(sentences_df)} sentences")

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
        text_sentences_df=text_sentences_df,
        text_artifacts_df=text_artifacts_df,
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


def _register_custom_gates(gates_config) -> dict:
    """Register config-defined gates; return stimulus path overrides."""
    from typo_eval.gates.prompts import GateSpec, register_gate_spec

    overrides = {}
    for name, cg in gates_config.custom_gates.items():
        register_gate_spec(GateSpec(
            gate=name,
            prompt_id=cg.prompt_id or f"{name}_custom",
            system_prompt=cg.system_prompt,
            question=cg.question,
            yes_is_favorable=cg.yes_is_favorable,
        ))
        if cg.stimulus_path:
            overrides[name] = cg.stimulus_path
    return overrides


def _gates_paths(gates_config) -> dict:
    """Standard paths for a gates run."""
    repo_root = get_repo_root()
    inputs_dir = repo_root / "data" / "inputs" / "gates"
    # Per-run metadata so configs with different variant sets don't clobber
    # each other; gates_v1 predates this and used the unsuffixed name.
    metadata_path = inputs_dir / f"metadata_{gates_config.run_tag}.csv"
    if not metadata_path.exists() and (inputs_dir / "metadata.csv").exists():
        fallback = inputs_dir / "metadata.csv"
    else:
        fallback = metadata_path
    return {
        "repo_root": repo_root,
        "inputs_dir": inputs_dir,
        "rendered_dir": repo_root / "data" / "rendered" / "gates",
        "metadata_path": metadata_path,
        "metadata_read_path": fallback,
        "run_dir": repo_root / "results" / "gates" / gates_config.run_tag,
    }


def cmd_gates_build(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Write gate stimulus CSVs."""
    from typo_eval.gates.config import load_gates_config
    from typo_eval.gates.stimuli import write_gate_csvs

    gc = load_gates_config(args.config)
    paths = _gates_paths(gc)
    df = write_gate_csvs(paths["inputs_dir"], gc.gates)
    logger.info(f"Wrote {len(df)} gate items to {paths['inputs_dir']}")
    for gate, n in df.groupby("gate").size().items():
        logger.info(f"  {gate}: {n} items")


def cmd_gates_render(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Render all gate items under all variants."""
    from typo_eval.gates.config import load_gates_config
    from typo_eval.gates.render import render_gate_items
    from typo_eval.gates.stimuli import load_gate_items

    gc = load_gates_config(args.config)
    overrides = _register_custom_gates(gc)
    paths = _gates_paths(gc)
    items_df = load_gate_items(paths["inputs_dir"], gc.gates, overrides)
    metadata = render_gate_items(gc, items_df, paths["rendered_dir"], paths["repo_root"])
    metadata.to_csv(paths["metadata_path"], index=False)
    logger.info(f"Rendered {len(metadata)} images; metadata at {paths['metadata_path']}")


def cmd_gates_calibrate(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run text-mode calibration for one provider, then summarize + select."""
    from typo_eval.gates.config import load_gates_config
    from typo_eval.gates.engine import (
        run_calibration,
        select_boundary_items,
        summarize_calibration,
    )
    from typo_eval.gates.stimuli import load_gate_items

    gc = load_gates_config(args.config)
    overrides = _register_custom_gates(gc)
    paths = _gates_paths(gc)
    run_dir = paths["run_dir"]
    logger = _setup_logger(run_dir)
    items_df = load_gate_items(paths["inputs_dir"], gc.gates, overrides)

    jsonl_path = run_calibration(
        gc, args.provider, items_df, run_dir,
        limit=args.limit, dry_run=args.dry_run,
    )
    if args.dry_run:
        return

    summary = summarize_calibration(jsonl_path)
    if not summary.empty:
        # Copied/merged JSONLs may carry gates outside this config
        summary = summary[summary["gate"].isin(gc.gates)]
    if summary.empty:
        logger.error("No calibration records found")
        sys.exit(1)
    cal_path = run_dir / "calibration" / f"summary_{args.provider}.csv"
    summary.to_csv(cal_path, index=False)

    selected = select_boundary_items(summary, gc)
    sel_dir = run_dir / "selection"
    sel_dir.mkdir(parents=True, exist_ok=True)
    sel_path = sel_dir / f"selected_{args.provider}.csv"
    selected.to_csv(sel_path, index=False)
    for gate, grp in selected.groupby("gate"):
        n_band = int(grp["in_band"].sum())
        logger.info(
            f"{gate}: selected {len(grp)} items ({n_band} in band), "
            f"p_yes range [{grp['p_yes'].min():.2f}, {grp['p_yes'].max():.2f}]"
        )
    logger.info(f"Selection written to {sel_path}")


def cmd_gates_run(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run the vision arm for one provider on its selected items."""
    import pandas as pd
    from typo_eval.gates.config import load_gates_config
    from typo_eval.gates.engine import run_vision

    gc = load_gates_config(args.config)
    _register_custom_gates(gc)
    paths = _gates_paths(gc)
    run_dir = paths["run_dir"]
    logger = _setup_logger(run_dir)

    sel_path = run_dir / "selection" / f"selected_{args.provider}.csv"
    if not sel_path.exists():
        logger.error(f"No selection file at {sel_path}; run gates-calibrate first")
        sys.exit(1)
    meta_path = paths["metadata_read_path"]
    if not meta_path.exists():
        logger.error(f"No render metadata at {meta_path}; run gates-render first")
        sys.exit(1)

    selected_df = pd.read_csv(sel_path)
    metadata_df = pd.read_csv(meta_path)
    run_vision(
        gc, args.provider, selected_df, metadata_df, run_dir,
        limit=args.limit, dry_run=args.dry_run,
    )


def cmd_gates_analyze(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Analyze a gates run."""
    from typo_eval.gates.analyze import analyze_gates_run
    from typo_eval.gates.config import load_gates_config

    gc = load_gates_config(args.config)
    _register_custom_gates(gc)
    paths = _gates_paths(gc)
    analysis_dir = analyze_gates_run(gc, paths["run_dir"])
    logger.info(f"Analysis complete: {analysis_dir}")


def cmd_gates_drift(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Compare two gates runs (model upgrade, prompt change, vendor swap)."""
    import datetime as _dt
    from typo_eval.gates.config import load_gates_config
    from typo_eval.gates.drift import compare_runs

    if args.config and Path(args.config).exists() and args.config.endswith((".yaml", ".yml")):
        try:
            gc = load_gates_config(args.config)
            _register_custom_gates(gc)
        except Exception:
            pass  # drift only needs custom-gate favorability if gates are custom

    repo_root = get_repo_root()
    gates_root = repo_root / "results" / "gates"

    def resolve(tag_or_path: str) -> Path:
        p = Path(tag_or_path)
        return p if p.exists() else gates_root / tag_or_path

    run_a, run_b = resolve(args.run_a), resolve(args.run_b)
    for p, flag in [(run_a, "--run-a"), (run_b, "--run-b")]:
        if not p.exists():
            logger.error(f"{flag}: no run at {p}")
            sys.exit(1)
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = gates_root / f"drift_{run_a.name}_vs_{run_b.name}_{stamp}"
    report = compare_runs(
        run_a, run_b, out_dir,
        label_a=args.run_a, label_b=args.run_b,
        provider=args.provider,
    )
    logger.info(f"Drift report: {report}")


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
    run_parser.add_argument(
        "--shard",
        help="Process only sentences where sentence_id %% N == i, as 'i/N'. "
        "Shards share a run-id and JSONL; keys never overlap across shards.",
    )

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze inference results")
    analyze_parser.add_argument("--run", dest="run_id", help="Run ID to analyze")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple runs across providers/models")
    compare_parser.add_argument("--run-ids", nargs="+", help="Specific run IDs to compare")
    compare_parser.add_argument("--provider", choices=["openai", "anthropic", "google"], help="Filter to specific provider")
    compare_parser.add_argument("--model", help="Filter to models containing this substring")

    # gates commands (decision-gate experiment; use a gates config YAML)
    subparsers.add_parser("gates-build", help="Write gate stimulus CSVs")
    subparsers.add_parser("gates-render", help="Render gate documents for all variants")
    for name, help_text in [
        ("gates-calibrate", "Text-mode calibration + boundary selection"),
        ("gates-run", "Vision arm on selected boundary items"),
    ]:
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument(
            "--provider",
            choices=["openai", "anthropic", "google"],
            required=True,
            help="Inference provider",
        )
        p.add_argument("--dry-run", action="store_true", help="Print planned calls without executing")
        p.add_argument("--limit", type=int, help="Limit number of inference calls")
    subparsers.add_parser("gates-analyze", help="Analyze a gates run")

    drift_parser = subparsers.add_parser(
        "gates-drift",
        help="Compare two gates runs (model upgrade / prompt change / vendor swap)",
    )
    drift_parser.add_argument("--run-a", required=True, help="Baseline run tag or path")
    drift_parser.add_argument("--run-b", required=True, help="Comparison run tag or path")
    drift_parser.add_argument(
        "--provider", choices=["openai", "anthropic", "google"],
        help="Restrict comparison to one provider",
    )

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

    # Gates commands use their own config schema; dispatch before loading
    # the v0 config.
    gates_commands = {
        "gates-build": cmd_gates_build,
        "gates-render": cmd_gates_render,
        "gates-calibrate": cmd_gates_calibrate,
        "gates-run": cmd_gates_run,
        "gates-analyze": cmd_gates_analyze,
        "gates-drift": cmd_gates_drift,
    }
    if args.command in gates_commands:
        gates_commands[args.command](args, logger)
        return

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
