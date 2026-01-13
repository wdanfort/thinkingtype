"""CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from typography.analysis import analyze_run, compare_runs
from typography.config import DEFAULT_CONFIG_PATH, TypographyConfig, load_config, resolve_paths
from typography.fontset import load_fontset, validate_fontset
from typography.generate import generate_artifacts
from typography.infer import infer, load_run_config
from typography.io import read_yaml


def _setup_logger(run_dir: Path | None) -> logging.Logger:
    logger = logging.getLogger("typography")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if run_dir:
        log_path = run_dir / "logs.txt"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _resolve_fontset_path(fontset_path: str) -> Path:
    candidate = Path(fontset_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def _derive_repo_root(fontset_path: Path) -> Path:
    if fontset_path.parent.name == "configs":
        return fontset_path.parent.parent
    return Path.cwd()


def _repo_root() -> Path:
    """Backward-compatible repo root helper for older entrypoints."""
    return Path.cwd()


def _load_fontset_or_exit(fontset_path: str, logger: logging.Logger) -> tuple[dict, Path]:
    resolved_path = _resolve_fontset_path(fontset_path)
    fontset = load_fontset(resolved_path)
    repo_root = _derive_repo_root(resolved_path)
    missing = validate_fontset(repo_root, fontset)
    fontset_id = fontset.get("fontset_id", "unknown")
    fonts_count = len(fontset.get("fonts", {}))
    logger.info("Fontset %s loaded (%d fonts).", fontset_id, fonts_count)
    if missing:
        missing_list = "\n".join(f" - {path}" for path in missing)
        raise SystemExit(
            "Missing font files for fontset "
            f"{fontset_id}:\n{missing_list}\n"
            "Place the files under assets/fonts/... or run scripts/fetch_fonts.py."
        )
    return fontset, repo_root


def _load_config_from_run(run_id: str, runs_root: str) -> TypographyConfig:
    run_dir = Path(runs_root) / run_id
    config_path = run_dir / "config_resolved.yaml"
    if config_path.exists():
        return TypographyConfig(**read_yaml(config_path))
    run_json = load_run_config(run_dir)
    config = load_config(DEFAULT_CONFIG_PATH)
    config.artifact_set_id = run_json["artifact_set_id"]
    config.inference.provider_text = run_json["provider_text"]
    config.inference.model_text = run_json["model_text"]
    config.inference.provider_image = run_json["provider_image"]
    config.inference.model_image = run_json["model_image"]
    config.inference.temperature = run_json["temperature"]
    return resolve_paths(config)


def main() -> None:
    load_dotenv()
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(prog="typography")
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    gen_parser.add_argument("--fontset", default="configs/fontset_v1_pinned_open_15.json")
    gen_parser.add_argument("--allow-system-fonts", action="store_true")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--config", default=None)
    infer_parser.add_argument("--run", dest="run_id", default=None)
    infer_parser.add_argument("--temperature", type=float, default=None)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--run", dest="run_id", default=None)
    analyze_parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument("--fontset", default="configs/fontset_v1_pinned_open_15.json")
    run_parser.add_argument("--allow-system-fonts", action="store_true")

    fonts_parser = subparsers.add_parser("fonts")
    fonts_parser.add_argument("--fontset", default="configs/fontset_v1_pinned_open_15.json")
    fonts_parser.add_argument("--check", action="store_true", help="Validate vendored font files exist.")

    args = parser.parse_args()

    if args.command == "generate":
        config = resolve_paths(load_config(args.config))
        logger = _setup_logger(None)
        fontset, repo_root = _load_fontset_or_exit(args.fontset, logger)
        generate_artifacts(config, logger, fontset, repo_root, allow_system_fonts=args.allow_system_fonts)
        return

    if args.command == "infer":
        if args.run_id:
            config = _load_config_from_run(args.run_id, "runs")
        else:
            config_path = args.config or str(DEFAULT_CONFIG_PATH)
            config = resolve_paths(load_config(config_path))
        run_dir = Path(config.output.runs_root) / (args.run_id or "")
        logger = _setup_logger(run_dir if run_dir.exists() else None)
        infer(config, args.run_id, args.temperature, logger)
        return

    if args.command == "analyze":
        if args.compare:
            run_a, run_b = args.compare
            logger = _setup_logger(None)
            compare_runs(run_a, run_b, "runs", "artifacts", logger)
            return
        if not args.run_id:
            raise SystemExit("--run is required for analyze without --compare")
        logger = _setup_logger(Path("runs") / args.run_id)
        analyze_run(args.run_id, "runs", "artifacts", logger)
        return

    if args.command == "run":
        config_path = args.config or str(DEFAULT_CONFIG_PATH)
        config = resolve_paths(load_config(config_path))
        logger = _setup_logger(None)
        fontset, repo_root = _load_fontset_or_exit(args.fontset, logger)
        generate_artifacts(config, logger, fontset, repo_root, allow_system_fonts=args.allow_system_fonts)
        run_id = infer(config, None, args.temperature, logger)
        analyze_run(run_id, config.output.runs_root, config.output.artifacts_root, logger)
        return

    if args.command == "fonts":
        logger = _setup_logger(None)
        _load_fontset_or_exit(args.fontset, logger)
        logger.info("Fontset check passed.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
