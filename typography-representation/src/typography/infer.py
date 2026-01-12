"""Inference loop."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
import logging
from typing import Dict, Iterable, Tuple

import pandas as pd
from typography.config import TypographyConfig, config_to_dict
from typography.io import append_csv, ensure_dir, read_json, write_json, write_yaml
from typography.metrics import parse_yes_no
from typography.providers import get_provider
from typography.schemas import DIMENSIONS, RESULTS_COLUMNS, InferenceRecord


def _temperature_token(temperature: float) -> str:
    return f"t{str(temperature).replace('.', 'p')}"


def build_run_id(config: TypographyConfig, temperature: float) -> str:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return (
        f"{config.artifact_set_id}_"
        f"{config.inference.provider_text}_{config.inference.model_text}_"
        f"{config.inference.provider_image}_{config.inference.model_image}_"
        f"{_temperature_token(temperature)}_{timestamp}"
    )


def load_existing_keys(results_path: Path) -> set[Tuple[str, ...]]:
    if not results_path.exists():
        return set()
    df = pd.read_csv(results_path)
    key_cols = [
        "run_id",
        "item_id",
        "variant_id",
        "representation",
        "dimension",
        "provider",
        "model",
        "temperature",
    ]
    if not set(key_cols).issubset(df.columns):
        return set()
    keys = set()
    for _, row in df[key_cols].iterrows():
        keys.add(tuple(str(row[col]) for col in key_cols))
    return keys


def _should_skip(existing_keys: set[Tuple[str, ...]], record: InferenceRecord) -> bool:
    key = (
        str(record.run_id),
        str(record.item_id),
        str(record.variant_id),
        str(record.representation),
        str(record.dimension),
        str(record.provider),
        str(record.model),
        str(record.temperature),
    )
    return key in existing_keys


def _retry_call(fn, max_retries: int, sleep_s: float):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise exc
            delay = sleep_s * (2**attempt)
            time.sleep(delay)
    return None


def _load_metadata(artifact_dir: Path) -> pd.DataFrame:
    metadata_path = artifact_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv at {metadata_path}")
    return pd.read_csv(metadata_path)


def infer(
    config: TypographyConfig,
    run_id: str | None,
    temperature_override: float | None,
    logger,
) -> str:
    temperature = temperature_override if temperature_override is not None else config.inference.temperature
    run_id = run_id or build_run_id(config, temperature)

    runs_root = Path(config.output.runs_root)
    run_dir = runs_root / run_id
    ensure_dir(run_dir)
    logs_path = run_dir / "logs.txt"
    if not logs_path.exists():
        logs_path.write_text("")
    if logger and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(logs_path)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    resolved_config = config_to_dict(config)
    resolved_config["inference"]["temperature"] = temperature

    run_json_path = run_dir / "run.json"
    if not run_json_path.exists():
        write_json(
            run_json_path,
            {
                "run_id": run_id,
                "artifact_set_id": config.artifact_set_id,
                "artifact_dir": str(Path(config.output_dir).resolve()),
                "provider_text": config.inference.provider_text,
                "model_text": config.inference.model_text,
                "provider_image": config.inference.provider_image,
                "model_image": config.inference.model_image,
                "temperature": temperature,
                "created_at": dt.datetime.utcnow().isoformat(),
            },
        )
        write_yaml(run_dir / "config_resolved.yaml", resolved_config)
    elif temperature_override is not None:
        run_json = read_json(run_json_path)
        run_json["temperature"] = temperature
        run_json["artifact_dir"] = str(Path(config.output_dir).resolve())
        write_json(run_json_path, run_json)
        write_yaml(run_dir / "config_resolved.yaml", resolved_config)

    artifact_dir = Path(config.output_dir)
    metadata = _load_metadata(artifact_dir)

    results_path = run_dir / "results.csv"
    existing_keys = load_existing_keys(results_path)

    provider_text = get_provider(config.inference.provider_text)
    provider_image = get_provider(config.inference.provider_image)

    def iter_text_rows() -> Iterable[dict]:
        text_rows = metadata[metadata["representation"] == "text"]
        seen = set()
        for _, row in text_rows.iterrows():
            item_id = str(row["item_id"])
            category = row["category"]
            text_path = row["text_path"]
            if item_id in seen:
                continue
            seen.add(item_id)
            input_text = Path(text_path).read_text()
            for dimension, question in DIMENSIONS.items():
                yield {
                    "item_id": item_id,
                    "category": category,
                    "text": input_text,
                    "dimension": dimension,
                    "question": question,
                }

    def iter_image_rows() -> Iterable[dict]:
        image_rows = metadata[metadata["representation"] == "image"]
        for _, row in image_rows.iterrows():
            item_id = str(row["item_id"])
            category = row["category"]
            variant_id = row["variant_id"]
            image_path = row["image_path"]
            for dimension, question in DIMENSIONS.items():
                yield {
                    "item_id": item_id,
                    "category": category,
                    "variant_id": variant_id,
                    "image_path": image_path,
                    "dimension": dimension,
                    "question": question,
                }

    rows_to_write = []

    def flush_rows():
        nonlocal rows_to_write
        if rows_to_write:
            append_csv(results_path, rows_to_write, RESULTS_COLUMNS)
            rows_to_write = []

    for row in iter_text_rows():
        record = InferenceRecord(
            run_id=run_id,
            artifact_set_id=config.artifact_set_id,
            provider=config.inference.provider_text,
            model=config.inference.model_text,
            temperature=temperature,
            item_id=row["item_id"],
            category=row["category"],
            variant_id="__text__",
            representation="text",
            dimension=row["dimension"],
            response_norm="",
            response_01=None,
            raw_response="",
            error=None,
            created_at=dt.datetime.utcnow().isoformat(),
        )
        if _should_skip(existing_keys, record):
            continue

        try:
            raw = _retry_call(
                lambda: provider_text.infer_text(
                    config.inference.model_text,
                    temperature,
                    row["text"],
                    row["question"],
                ),
                max_retries=config.inference.max_retries,
                sleep_s=config.inference.rate_limit_sleep,
            )
            parsed, norm = parse_yes_no(raw)
            record = record.__class__(
                **{**record.__dict__, "response_norm": norm, "response_01": parsed, "raw_response": raw}
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Text inference error for item %s dim %s: %s", row["item_id"], row["dimension"], exc)
            record = record.__class__(**{**record.__dict__, "error": str(exc)})
            if config.inference.fail_fast:
                raise

        rows_to_write.append(record.to_row())
        existing_keys.add(
            (
                str(record.run_id),
                str(record.item_id),
                str(record.variant_id),
                str(record.representation),
                str(record.dimension),
                str(record.provider),
                str(record.model),
                str(record.temperature),
            )
        )
        if len(rows_to_write) >= 25:
            flush_rows()

    flush_rows()

    for row in iter_image_rows():
        record = InferenceRecord(
            run_id=run_id,
            artifact_set_id=config.artifact_set_id,
            provider=config.inference.provider_image,
            model=config.inference.model_image,
            temperature=temperature,
            item_id=row["item_id"],
            category=row["category"],
            variant_id=row["variant_id"],
            representation="image",
            dimension=row["dimension"],
            response_norm="",
            response_01=None,
            raw_response="",
            error=None,
            created_at=dt.datetime.utcnow().isoformat(),
        )
        if _should_skip(existing_keys, record):
            continue

        try:
            raw = _retry_call(
                lambda: provider_image.infer_image(
                    config.inference.model_image,
                    temperature,
                    row["image_path"],
                    row["question"],
                ),
                max_retries=config.inference.max_retries,
                sleep_s=config.inference.rate_limit_sleep,
            )
            parsed, norm = parse_yes_no(raw)
            record = record.__class__(
                **{**record.__dict__, "response_norm": norm, "response_01": parsed, "raw_response": raw}
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Image inference error for item %s variant %s dim %s: %s", row["item_id"], row["variant_id"], row["dimension"], exc)
            record = record.__class__(**{**record.__dict__, "error": str(exc)})
            if config.inference.fail_fast:
                raise

        rows_to_write.append(record.to_row())
        existing_keys.add(
            (
                str(record.run_id),
                str(record.item_id),
                str(record.variant_id),
                str(record.representation),
                str(record.dimension),
                str(record.provider),
                str(record.model),
                str(record.temperature),
            )
        )
        if len(rows_to_write) >= 25:
            flush_rows()

    flush_rows()
    _update_registry(run_id, config, temperature, run_dir, results_path, logger)
    return run_id


def _update_registry(
    run_id: str,
    config: TypographyConfig,
    temperature: float,
    run_dir: Path,
    results_path: Path,
    logger,
) -> None:
    registry_path = Path(config.output.runs_root) / "index.csv"
    record = {
        "run_id": run_id,
        "created_at": dt.datetime.utcnow().isoformat(),
        "artifact_set_id": config.artifact_set_id,
        "provider_text": config.inference.provider_text,
        "model_text": config.inference.model_text,
        "provider_image": config.inference.provider_image,
        "model_image": config.inference.model_image,
        "temperature": temperature,
        "config_path": str(run_dir / "config_resolved.yaml"),
        "results_path": str(results_path),
        "analysis_dir": "",
        "git_commit": _git_commit_hash(),
        "notes": "",
    }

    if registry_path.exists():
        existing = pd.read_csv(registry_path)
        if (existing["run_id"] == run_id).any():
            logger.info("Run %s already in registry", run_id)
            return

    append_csv(
        registry_path,
        [record],
        [
            "run_id",
            "created_at",
            "artifact_set_id",
            "provider_text",
            "model_text",
            "provider_image",
            "model_image",
            "temperature",
            "config_path",
            "results_path",
            "analysis_dir",
            "git_commit",
            "notes",
        ],
    )


def _git_commit_hash() -> str:
    git_head = Path(".git/HEAD")
    if not git_head.exists():
        return ""
    ref = git_head.read_text().strip()
    if ref.startswith("ref:"):
        ref_path = Path(".git") / ref.replace("ref: ", "")
        if ref_path.exists():
            return ref_path.read_text().strip()
    return ref


def load_run_config(run_dir: Path) -> Dict[str, str]:
    run_json_path = run_dir / "run.json"
    if not run_json_path.exists():
        raise FileNotFoundError(f"Missing run.json at {run_json_path}")
    return read_json(run_json_path)
