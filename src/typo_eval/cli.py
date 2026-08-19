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


def _hrsim_paths(hc) -> dict:
    """Standard paths for an hrsim run."""
    repo_root = get_repo_root()
    inputs_dir = repo_root / "data" / "inputs" / "hrsim"
    return {
        "repo_root": repo_root,
        "inputs_dir": inputs_dir,
        "items_path": inputs_dir / f"items_{hc.run_tag}.csv",
        "rendered_dir": repo_root / "data" / "rendered" / hc.run_tag,
        "manifest_path": inputs_dir / f"render_manifest_{hc.run_tag}.csv",
        "run_dir": repo_root / "results" / "hrsim" / hc.run_tag,
    }


def cmd_hrsim_build(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Sample real resumes, pseudonymize, and pair with junior/senior JDs."""
    from typo_eval.hrsim.build import build_items, write_items_csv
    from typo_eval.hrsim.config import load_hrsim_config

    hc = load_hrsim_config(args.config)
    paths = _hrsim_paths(hc)
    resumes_csv = paths["repo_root"] / "data" / "inputs" / "resumes_raw" / "Resume.csv"
    if not resumes_csv.exists():
        logger.error(f"Resume corpus not found: {resumes_csv}")
        sys.exit(1)

    df = build_items(hc, resumes_csv)
    write_items_csv(df, paths["items_path"])
    logger.info(f"Wrote {len(df)} hrsim items to {paths['items_path']}")
    for (category, jd_level), n in df.groupby(["category", "jd_level"]).size().items():
        logger.info(f"  {category} / {jd_level}: {n} items")


def cmd_hrsim_render(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Render every sampled resume under every font variant."""
    import pandas as pd
    from typo_eval.hrsim.config import load_hrsim_config
    from typo_eval.hrsim.render import render_hrsim_resumes

    hc = load_hrsim_config(args.config)
    paths = _hrsim_paths(hc)
    if not paths["items_path"].exists():
        logger.error(f"No items at {paths['items_path']}; run hrsim-build first")
        sys.exit(1)
    items_df = pd.read_csv(paths["items_path"])
    resumes_df = items_df[["resume_id", "resume_text"]].drop_duplicates(subset=["resume_id"])

    manifest = render_hrsim_resumes(
        hc, resumes_df, paths["rendered_dir"], paths["repo_root"],
        limit=getattr(args, "limit", None),
    )
    manifest.to_csv(paths["manifest_path"], index=False)
    logger.info(
        f"Rendered {len(manifest)} images ({manifest['resume_id'].nunique()} resumes); "
        f"manifest at {paths['manifest_path']}"
    )
    over = manifest[manifest["image_height"] > hc.render.max_image_height]
    if not over.empty:
        logger.warning(f"{len(over)} renders still exceed max_image_height after retries")
    logger.info(
        f"Image height range: [{manifest['image_height'].min()}, "
        f"{manifest['image_height'].max()}]"
    )


def cmd_hrsim_run(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run one model leg (text arm + image arm x all variants)."""
    import pandas as pd
    from typo_eval.hrsim.config import load_hrsim_config
    from typo_eval.hrsim.engine import run_leg

    hc = load_hrsim_config(args.config)
    paths = _hrsim_paths(hc)
    run_dir = paths["run_dir"]
    logger = _setup_logger(run_dir)

    if not paths["items_path"].exists():
        logger.error(f"No items at {paths['items_path']}; run hrsim-build first")
        sys.exit(1)
    if not paths["manifest_path"].exists():
        logger.error(f"No render manifest at {paths['manifest_path']}; run hrsim-render first")
        sys.exit(1)

    items_df = pd.read_csv(paths["items_path"])
    manifest_df = pd.read_csv(paths["manifest_path"])
    run_leg(
        hc, args.leg, items_df, manifest_df, run_dir,
        limit_items=getattr(args, "limit_items", None),
        limit=getattr(args, "limit", None),
        dry_run=getattr(args, "dry_run", False),
    )


def cmd_hrsim_analyze(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Analyze an hrsim run: accept rates, deltas, flips, reasoning mentions."""
    from typo_eval.hrsim.analyze import analyze_hrsim_run
    from typo_eval.hrsim.config import load_hrsim_config

    hc = load_hrsim_config(args.config)
    paths = _hrsim_paths(hc)
    analysis_dir = analyze_hrsim_run(paths["run_dir"])
    logger.info(f"Analysis complete: {analysis_dir}")


def _hrsim2_paths(hc) -> dict:
    """Standard paths for an hrsim2 run. Parallels _hrsim_paths but under
    its own fixed input dirname (hrsim2, not hrsim) and its own rendered/
    results locations -- v1's are never touched."""
    repo_root = get_repo_root()
    inputs_dir = repo_root / "data" / "inputs" / "hrsim2"
    return {
        "repo_root": repo_root,
        "inputs_dir": inputs_dir,
        "resumes_csv": repo_root / "data" / "inputs" / "resumes_raw" / "Resume.csv",
        "jd_csv": repo_root / "data" / "inputs" / "jds_raw" / "JD_data.csv",
        "crosswalk_audit_path": inputs_dir / "crosswalk_audit.md",
        "items_path": inputs_dir / f"items_{hc.run_tag}.csv",
        "interaction_items_path": inputs_dir / f"items_interaction_{hc.run_tag}.csv",
        "rendered_dir": repo_root / "data" / "rendered" / hc.run_tag,
        "interaction_rendered_dir": repo_root / "data" / "rendered" / "hrsim2_interaction",
        "manifest_path": inputs_dir / f"render_manifest_{hc.run_tag}.csv",
        "interaction_manifest_path": inputs_dir / f"render_manifest_interaction_{hc.run_tag}.csv",
        "run_dir": repo_root / "results" / "hrsim" / hc.run_tag,
    }


def cmd_hrsim2_build(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Validate the JD crosswalk against the real corpus, sample resumes,
    and pair each with 2 seeded-sampled real JDs (+ pick 1 for the
    interaction sub-study)."""
    from typo_eval.hrsim2 import crosswalk
    from typo_eval.hrsim2.build import build_items, write_items_csv
    from typo_eval.hrsim2.config import load_hrsim_v2_config
    from typo_eval.hrsim.build import _EMAIL_RE, _PHONE_RE, _URL_RE

    hc = load_hrsim_v2_config(args.config)
    paths = _hrsim2_paths(hc)
    if not paths["resumes_csv"].exists():
        logger.error(f"Resume corpus not found: {paths['resumes_csv']}")
        sys.exit(1)

    jd_csv = crosswalk.ensure_jd_corpus_copied(paths["repo_root"])
    jd_df = crosswalk.load_jd_raw(jd_csv)
    pools = crosswalk.build_all_pools(jd_df)
    try:
        crosswalk.validate_pools(pools)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    report = crosswalk.write_crosswalk_audit(jd_df, pools, paths["crosswalk_audit_path"])
    print(report)
    logger.info(f"Crosswalk audit written to {paths['crosswalk_audit_path']}")

    items_df, interaction_df = build_items(hc, paths["resumes_csv"], pools, paths["repo_root"])
    write_items_csv(items_df, paths["items_path"])
    write_items_csv(interaction_df, paths["interaction_items_path"])
    logger.info(f"Wrote {len(items_df)} hrsim2 items to {paths['items_path']}")
    logger.info(
        f"Wrote {len(interaction_df)} interaction items to {paths['interaction_items_path']}"
    )

    n_occ = items_df["category"].nunique()
    n_resumes = items_df["resume_id"].nunique()
    expected = n_occ * hc.sampling.per_category * hc.jds_per_resume
    logger.info(
        f"{len(items_df)} items = {n_occ} occupations x {n_resumes} resumes x "
        f"{hc.jds_per_resume} draws (expected {expected})"
    )
    if len(items_df) != expected:
        logger.error(f"Item count mismatch: got {len(items_df)}, expected {expected}")
        sys.exit(1)

    residual = 0
    for name, pattern in [("email", _EMAIL_RE), ("url", _URL_RE), ("phone", _PHONE_RE)]:
        n = int(items_df["resume_text"].str.contains(pattern).sum())
        logger.info(f"Residual {name} matches: {n}")
        residual += n
    if residual:
        logger.error(f"{residual} residual PII matches survived the scrub")
        sys.exit(1)

    sample = items_df.iloc[0]
    logger.info(
        f"Sample item: occ={sample['category']} jd_job={sample['jd_job']!r} "
        f"resume_text[:200]={sample['resume_text'][:200]!r}"
    )


def cmd_hrsim2_render(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Render every sampled resume under every font variant (main run) plus
    the name x font interaction sub-study's renders (different header)."""
    import pandas as pd
    from typo_eval.hrsim2.config import load_hrsim_v2_config
    from typo_eval.hrsim2.render import render_interaction, render_main

    hc = load_hrsim_v2_config(args.config)
    paths = _hrsim2_paths(hc)
    if not paths["items_path"].exists():
        logger.error(f"No items at {paths['items_path']}; run hrsim2-build first")
        sys.exit(1)
    if not paths["interaction_items_path"].exists():
        logger.error(
            f"No interaction items at {paths['interaction_items_path']}; run hrsim2-build first"
        )
        sys.exit(1)

    items_df = pd.read_csv(paths["items_path"])
    manifest = render_main(
        hc, items_df, paths["rendered_dir"], paths["repo_root"],
        limit=getattr(args, "limit", None),
    )
    manifest.to_csv(paths["manifest_path"], index=False)
    logger.info(
        f"Rendered {len(manifest)} main images ({manifest['resume_id'].nunique()} resumes); "
        f"manifest at {paths['manifest_path']}"
    )
    over = manifest[manifest["image_height"] > hc.render.max_image_height]
    if not over.empty:
        logger.warning(f"{len(over)} main renders still exceed max_image_height after retries")
    logger.info(
        f"Main image height range: [{manifest['image_height'].min()}, "
        f"{manifest['image_height'].max()}]"
    )

    interaction_df = pd.read_csv(paths["interaction_items_path"])
    inter_manifest = render_interaction(
        hc, interaction_df, paths["interaction_rendered_dir"], paths["repo_root"],
        limit=getattr(args, "limit", None),
    )
    inter_manifest.to_csv(paths["interaction_manifest_path"], index=False)
    logger.info(
        f"Rendered {len(inter_manifest)} interaction images "
        f"({inter_manifest['resume_id'].nunique()} resumes); "
        f"manifest at {paths['interaction_manifest_path']}"
    )
    over_i = inter_manifest[inter_manifest["image_height"] > hc.render.max_image_height]
    if not over_i.empty:
        logger.warning(f"{len(over_i)} interaction renders still exceed max_image_height after retries")
    if not inter_manifest.empty:
        logger.info(
            f"Interaction image height range: [{inter_manifest['image_height'].min()}, "
            f"{inter_manifest['image_height'].max()}]"
        )


def cmd_hrsim4_run(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run one hrsim4 leg over the boundary-calibrated item set."""
    from typo_eval.hrsim4 import prompts
    from typo_eval.hrsim4.config import load_hrsim_v4_config
    from typo_eval.hrsim4.engine import build_items, run_leg

    repo = Path(__file__).resolve().parents[2]
    hc = load_hrsim_v4_config(args.config)

    # The 40-60% accept band and the boundary-sensitivity gate were measured
    # under the calibration script's exact wording. Refuse to run if it drifted.
    prompts.assert_matches_calibration(repo)

    run_dir = repo / hc.calibration_dir
    logger = _setup_logger(run_dir)

    legs = {leg.name: leg for leg in hc.legs}
    if args.leg not in legs:
        logger.error(f"Unknown leg {args.leg!r}; config has {sorted(legs)}")
        sys.exit(1)

    items = build_items(hc, repo)
    if getattr(args, "limit_items", None):
        renders = items.attrs["renders"]
        items = items.head(args.limit_items).copy()
        items.attrs["renders"] = renders

    cells = len(items) * len(hc.arms()) * hc.repeats
    logger.info(
        f"hrsim4 {args.leg}: {len(items)} items x {len(hc.arms())} arms x "
        f"{hc.repeats} repeats = {cells} cells "
        f"({int(items.boundary_sensitive.sum())} items boundary-sensitive)"
    )
    if getattr(args, "dry_run", False):
        logger.info("dry run -- nothing executed")
        return

    df = run_leg(hc, legs[args.leg], items, run_dir)
    ok = int(df.parsed_decision.notna().sum()) if len(df) else 0
    logger.info(f"leg complete: {ok}/{len(df)} records with a parsed decision")


def cmd_hrsim4_analyze(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Analyze an hrsim4 run: modality, typeface, and ink-ladder axes."""
    from typo_eval.hrsim4.analyze import analyze_run
    from typo_eval.hrsim4.config import load_hrsim_v4_config

    repo = Path(__file__).resolve().parents[2]
    hc = load_hrsim_v4_config(args.config)
    run_dir = repo / hc.calibration_dir
    tables = analyze_run(run_dir, repo / hc.corpus_dir, run_dir / "analysis")
    for name in ("completeness", "accept_rates", "modality_effects",
                 "typeface_effects", "ink_slope", "sensitivity_check",
                 "typography_mentions"):
        t = tables.get(name)
        if t is not None and len(t):
            print(f"\n=== {name} ===")
            print(t.to_string(index=False))


def cmd_hrsim2_run(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run one model leg: main phase (name_race="_ambiguous_", 6 arms) and/or
    the interaction phase (name_race in {white, black}, 3 arms)."""
    import pandas as pd
    from typo_eval.hrsim2.config import load_hrsim_v2_config
    from typo_eval.hrsim2.engine import run_leg

    hc = load_hrsim_v2_config(args.config)
    paths = _hrsim2_paths(hc)
    run_dir = paths["run_dir"]
    logger = _setup_logger(run_dir)

    required = ["items_path", "manifest_path", "interaction_items_path", "interaction_manifest_path"]
    for key in required:
        if not paths[key].exists():
            logger.error(f"Missing {paths[key]}; run hrsim2-build and hrsim2-render first")
            sys.exit(1)

    items_df = pd.read_csv(paths["items_path"])
    manifest_df = pd.read_csv(paths["manifest_path"])
    interaction_df = pd.read_csv(paths["interaction_items_path"])
    interaction_manifest_df = pd.read_csv(paths["interaction_manifest_path"])

    run_leg(
        hc, args.leg, items_df, manifest_df, interaction_df, interaction_manifest_df, run_dir,
        phase=getattr(args, "phase", "both"),
        limit_items=getattr(args, "limit_items", None),
        limit=getattr(args, "limit", None),
        dry_run=getattr(args, "dry_run", False),
    )


def cmd_hrsim2_analyze(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Analyze an hrsim2 run: main-phase accept rates/deltas, seniority
    moderation, the name x font interaction contrasts, flips, and mention
    scans."""
    from typo_eval.hrsim2.analyze import analyze_hrsim2_run
    from typo_eval.hrsim2.config import load_hrsim_v2_config

    hc = load_hrsim_v2_config(args.config)
    paths = _hrsim2_paths(hc)
    analysis_dir = analyze_hrsim2_run(paths["run_dir"])
    logger.info(f"Analysis complete: {analysis_dir}")


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
        choices=["openai", "anthropic", "google", "openai_compatible", "kimi"],
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
    compare_parser.add_argument("--provider", choices=["openai", "anthropic", "google", "openai_compatible", "kimi"], help="Filter to specific provider")
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
            choices=["openai", "anthropic", "google", "openai_compatible", "kimi"],
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
        "--provider", choices=["openai", "anthropic", "google", "openai_compatible", "kimi"],
        help="Restrict comparison to one provider",
    )

    # hrsim commands (resume x typography HR-simulation experiment; use an
    # hrsim config YAML, e.g. configs/hrsim_v1.yaml)
    subparsers.add_parser("hrsim-build", help="Sample resumes, pseudonymize, pair with JDs")
    hrsim_render_parser = subparsers.add_parser(
        "hrsim-render", help="Render sampled resumes under all font variants"
    )
    hrsim_render_parser.add_argument(
        "--limit", type=int, help="Render only the first N distinct resumes"
    )
    hrsim_run_parser = subparsers.add_parser(
        "hrsim-run", help="Run one model leg (text arm + image arm x variants)"
    )
    hrsim_run_parser.add_argument("--leg", required=True, help="Leg name from the config's legs list")
    hrsim_run_parser.add_argument("--dry-run", action="store_true", help="Print planned calls without executing")
    hrsim_run_parser.add_argument("--limit", type=int, help="Cap the raw number of API calls")
    hrsim_run_parser.add_argument(
        "--limit-items", type=int,
        help="Restrict to the first N distinct item_ids (all arms still run for those items)",
    )
    subparsers.add_parser("hrsim-analyze", help="Analyze an hrsim run")

    # hrsim2 commands (v2: real JDs via a curated crosswalk + a name x font
    # interaction sub-study; use configs/hrsim_v2.yaml). Parallel, separate
    # pipeline from hrsim -- does not touch hrsim/ or results/hrsim/hrsim_v1.
    subparsers.add_parser(
        "hrsim2-build", help="Validate JD crosswalk, sample resumes, pair with real JDs"
    )
    hrsim2_render_parser = subparsers.add_parser(
        "hrsim2-render", help="Render main + name x font interaction resumes"
    )
    hrsim2_render_parser.add_argument(
        "--limit", type=int,
        help="Render only the first N distinct resumes (applies to both main and interaction)",
    )
    hrsim2_run_parser = subparsers.add_parser(
        "hrsim2-run", help="Run one model leg (main phase + interaction phase)"
    )
    hrsim2_run_parser.add_argument("--leg", required=True, help="Leg name from the config's legs list")
    hrsim2_run_parser.add_argument(
        "--phase", choices=["main", "interaction", "both"], default="both",
        help="Which run phase(s) to execute",
    )
    hrsim2_run_parser.add_argument("--dry-run", action="store_true", help="Print planned calls without executing")
    hrsim2_run_parser.add_argument("--limit", type=int, help="Cap the raw number of API calls")
    hrsim2_run_parser.add_argument(
        "--limit-items", type=int,
        help="Restrict to the first N distinct item_ids per phase (all arms/name_races still run)",
    )
    subparsers.add_parser("hrsim2-analyze", help="Analyze an hrsim2 run")

    # hrsim4: corpus and JD calibration are built by the loop scripts, so the
    # CLI only runs and analyzes.
    hrsim4_run_parser = subparsers.add_parser(
        "hrsim4-run", help="Run one hrsim4 model leg over the calibrated item set"
    )
    hrsim4_run_parser.add_argument("--leg", required=True, help="Leg name from the config's legs list")
    hrsim4_run_parser.add_argument(
        "--limit-items", type=int,
        help="Restrict to the first N items (all arms still run) -- for smoke tests",
    )
    hrsim4_run_parser.add_argument("--dry-run", action="store_true", help="Plan and cost without executing")
    subparsers.add_parser("hrsim4-analyze", help="Analyze an hrsim4 run")

    # all command
    all_parser = subparsers.add_parser("all", help="Run full pipeline")
    all_parser.add_argument("--dry-run", action="store_true", help="Print planned calls without executing")
    all_parser.add_argument("--limit", type=int, help="Limit number of inference calls")
    all_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google", "openai_compatible", "kimi"],
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

    # hrsim commands use their own config schema too; dispatch before
    # loading the v0 config.
    hrsim_commands = {
        "hrsim-build": cmd_hrsim_build,
        "hrsim-render": cmd_hrsim_render,
        "hrsim-run": cmd_hrsim_run,
        "hrsim-analyze": cmd_hrsim_analyze,
    }
    if args.command in hrsim_commands:
        hrsim_commands[args.command](args, logger)
        return

    # hrsim2 commands use their own config schema too; dispatch before
    # loading the v0 config.
    hrsim2_commands = {
        "hrsim2-build": cmd_hrsim2_build,
        "hrsim2-render": cmd_hrsim2_render,
        "hrsim2-run": cmd_hrsim2_run,
        "hrsim2-analyze": cmd_hrsim2_analyze,
        "hrsim4-run": cmd_hrsim4_run,
        "hrsim4-analyze": cmd_hrsim4_analyze,
    }
    if args.command in hrsim2_commands:
        hrsim2_commands[args.command](args, logger)
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
