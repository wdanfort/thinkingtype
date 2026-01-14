"""Inference loop with JSONL logging and resume support."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from typo_eval.config import TypoEvalConfig
from typo_eval.prompts import (
    DIMENSIONS,
    DIMENSIONS_SYSTEM_PROMPT,
    DECISION_SYSTEM_PROMPT,
    DECISION_PROMPTS,
    compute_prompt_hash,
    parse_yes_no,
)
from typo_eval.providers import get_provider

logger = logging.getLogger(__name__)


@dataclass
class InferenceRecord:
    """Canonical record schema for JSONL logging."""

    run_id: str
    timestamp: str
    provider: str
    model: str
    temperature: float
    mode: str  # "dimensions" or "decision"
    representation: str  # "ocr" or "image"
    input_type: str  # "sentence" or "artifact"
    sentence_id: Optional[int] = None
    artifact_id: Optional[str] = None
    variant_id: str = "__ocr__"  # "__ocr__" for OCR baseline
    dimension_id: Optional[str] = None
    prompt_id: str = ""
    prompt_hash: str = ""
    input_hash: str = ""
    raw_response: str = ""
    parsed_response: Optional[int] = None
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def get_key(self) -> Tuple:
        """Get unique key for deduplication."""
        if self.mode == "dimensions":
            return (
                self.mode,
                self.representation,
                self.input_type,
                self.sentence_id or self.artifact_id,
                self.variant_id,
                self.dimension_id,
                self.provider,
                self.model,
                self.temperature,
                self.prompt_hash,
            )
        else:  # decision
            return (
                self.mode,
                self.representation,
                self.input_type,
                self.sentence_id or self.artifact_id,
                self.variant_id,
                self.provider,
                self.model,
                self.temperature,
                self.prompt_hash,
            )


def load_existing_keys(jsonl_path: Path) -> Set[Tuple]:
    """Load existing record keys from JSONL for resume support."""
    keys: Set[Tuple] = set()
    if not jsonl_path.exists():
        return keys

    with jsonl_path.open("r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                record = InferenceRecord(**data)
                keys.add(record.get_key())
            except (json.JSONDecodeError, TypeError):
                continue

    return keys


def append_jsonl(path: Path, record: InferenceRecord) -> None:
    """Append record to JSONL file (atomic write with flush)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")
        f.flush()


def compute_input_hash(content: str | bytes) -> str:
    """Compute hash of input content."""
    if isinstance(content, str):
        content = content.encode()
    return hashlib.sha256(content).hexdigest()[:16]


def _retry_call(fn, max_retries: int, sleep_s: float, jitter: float = 0.25):
    """Retry function with exponential backoff."""
    import random

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise exc
            delay = sleep_s * (2 ** attempt) * (1 + random.uniform(-jitter, jitter))
            logger.warning(f"Retry {attempt + 1}/{max_retries}: {exc}, sleeping {delay:.1f}s")
            time.sleep(delay)
    return None


def run_inference(
    config: TypoEvalConfig,
    run_id: str,
    run_dir: Path,
    sentences_df: Optional[pd.DataFrame] = None,
    artifacts_df: Optional[pd.DataFrame] = None,
    ocr_sentences_df: Optional[pd.DataFrame] = None,
    ocr_artifacts_df: Optional[pd.DataFrame] = None,
    sentences_metadata_df: Optional[pd.DataFrame] = None,
    artifacts_metadata_df: Optional[pd.DataFrame] = None,
    provider_name: str = "openai",
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> Path:
    """
    Run inference loop for dimensions and/or decision modes.

    Writes results to JSONL file with resume support.
    """
    jsonl_path = run_dir / "raw" / "responses.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing keys for resume
    existing_keys = load_existing_keys(jsonl_path)
    logger.info(f"Loaded {len(existing_keys)} existing records for resume")

    # Get provider config
    provider_config = getattr(config.providers, provider_name)
    provider = get_provider(provider_name)

    inference_cfg = config.inference
    temperature = inference_cfg.temperature
    mode = inference_cfg.mode

    total_calls = 0
    skipped_calls = 0

    # Define what to run
    run_dimensions = mode in ("dimensions", "both")
    run_decision = mode in ("decision", "both")

    # Process sentences
    if sentences_df is not None and config.inputs.sentences.get("enabled", True):
        # OCR baseline (deduped - one per sentence)
        if ocr_sentences_df is not None:
            for _, row in tqdm(
                ocr_sentences_df.iterrows(),
                total=len(ocr_sentences_df),
                desc="OCR inference (sentences)",
            ):
                sentence_id = int(row["sentence_id"])
                ocr_text = row["ocr_text"]
                input_hash = compute_input_hash(ocr_text)

                if limit and total_calls >= limit:
                    break

                # Dimensions mode - OCR
                if run_dimensions:
                    for dim_id in inference_cfg.dimensions:
                        question = DIMENSIONS.get(dim_id, "")
                        prompt_hash = compute_prompt_hash(DIMENSIONS_SYSTEM_PROMPT, question)

                        record = InferenceRecord(
                            run_id=run_id,
                            timestamp=dt.datetime.utcnow().isoformat(),
                            provider=provider_name,
                            model=provider_config.model_text,
                            temperature=temperature,
                            mode="dimensions",
                            representation="ocr",
                            input_type="sentence",
                            sentence_id=sentence_id,
                            variant_id="__ocr__",
                            dimension_id=dim_id,
                            prompt_id=dim_id,
                            prompt_hash=prompt_hash,
                            input_hash=input_hash,
                        )

                        if record.get_key() in existing_keys:
                            skipped_calls += 1
                            continue

                        total_calls += 1
                        if dry_run:
                            continue

                        try:
                            raw = _retry_call(
                                lambda: provider.infer_text(
                                    provider_config.model_text,
                                    temperature,
                                    ocr_text,
                                    question,
                                    DIMENSIONS_SYSTEM_PROMPT,
                                ),
                                max_retries=inference_cfg.max_retries,
                                sleep_s=inference_cfg.rate_limit_sleep,
                            )
                            parsed, _ = parse_yes_no(raw)
                            record.raw_response = raw
                            record.parsed_response = parsed
                        except Exception as exc:
                            record.parse_error = str(exc)
                            if inference_cfg.fail_fast:
                                raise

                        append_jsonl(jsonl_path, record)
                        existing_keys.add(record.get_key())

                # Decision mode - OCR
                if run_decision:
                    decision_prompt_id = inference_cfg.decision_prompt_id
                    question = DECISION_PROMPTS.get(decision_prompt_id, "")
                    prompt_hash = compute_prompt_hash(DECISION_SYSTEM_PROMPT, question)

                    record = InferenceRecord(
                        run_id=run_id,
                        timestamp=dt.datetime.utcnow().isoformat(),
                        provider=provider_name,
                        model=provider_config.model_text,
                        temperature=temperature,
                        mode="decision",
                        representation="ocr",
                        input_type="sentence",
                        sentence_id=sentence_id,
                        variant_id="__ocr__",
                        prompt_id=decision_prompt_id,
                        prompt_hash=prompt_hash,
                        input_hash=input_hash,
                    )

                    if record.get_key() in existing_keys:
                        skipped_calls += 1
                        continue

                    total_calls += 1
                    if dry_run:
                        continue

                    try:
                        raw = _retry_call(
                            lambda: provider.infer_text(
                                provider_config.model_text,
                                temperature,
                                ocr_text,
                                question,
                                DECISION_SYSTEM_PROMPT,
                            ),
                            max_retries=inference_cfg.max_retries,
                            sleep_s=inference_cfg.rate_limit_sleep,
                        )
                        parsed, _ = parse_yes_no(raw)
                        record.raw_response = raw
                        record.parsed_response = parsed
                    except Exception as exc:
                        record.parse_error = str(exc)
                        if inference_cfg.fail_fast:
                            raise

                    append_jsonl(jsonl_path, record)
                    existing_keys.add(record.get_key())

        # Image variants
        if sentences_metadata_df is not None:
            for _, row in tqdm(
                sentences_metadata_df.iterrows(),
                total=len(sentences_metadata_df),
                desc="Image inference (sentences)",
            ):
                sentence_id = int(row["sentence_id"])
                variant_id = row["variant_id"]
                image_path = row["image_path"]

                if not Path(image_path).exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue

                input_hash = compute_input_hash(Path(image_path).read_bytes())

                if limit and total_calls >= limit:
                    break

                # Dimensions mode - Image
                if run_dimensions:
                    for dim_id in inference_cfg.dimensions:
                        question = DIMENSIONS.get(dim_id, "")
                        prompt_hash = compute_prompt_hash(DIMENSIONS_SYSTEM_PROMPT, question)

                        record = InferenceRecord(
                            run_id=run_id,
                            timestamp=dt.datetime.utcnow().isoformat(),
                            provider=provider_name,
                            model=provider_config.model_vision,
                            temperature=temperature,
                            mode="dimensions",
                            representation="image",
                            input_type="sentence",
                            sentence_id=sentence_id,
                            variant_id=variant_id,
                            dimension_id=dim_id,
                            prompt_id=dim_id,
                            prompt_hash=prompt_hash,
                            input_hash=input_hash,
                        )

                        if record.get_key() in existing_keys:
                            skipped_calls += 1
                            continue

                        total_calls += 1
                        if dry_run:
                            continue

                        try:
                            raw = _retry_call(
                                lambda: provider.infer_image(
                                    provider_config.model_vision,
                                    temperature,
                                    image_path,
                                    question,
                                    DIMENSIONS_SYSTEM_PROMPT,
                                ),
                                max_retries=inference_cfg.max_retries,
                                sleep_s=inference_cfg.rate_limit_sleep,
                            )
                            parsed, _ = parse_yes_no(raw)
                            record.raw_response = raw
                            record.parsed_response = parsed
                        except Exception as exc:
                            record.parse_error = str(exc)
                            if inference_cfg.fail_fast:
                                raise

                        append_jsonl(jsonl_path, record)
                        existing_keys.add(record.get_key())

                # Decision mode - Image
                if run_decision:
                    decision_prompt_id = inference_cfg.decision_prompt_id
                    question = DECISION_PROMPTS.get(decision_prompt_id, "")
                    prompt_hash = compute_prompt_hash(DECISION_SYSTEM_PROMPT, question)

                    record = InferenceRecord(
                        run_id=run_id,
                        timestamp=dt.datetime.utcnow().isoformat(),
                        provider=provider_name,
                        model=provider_config.model_vision,
                        temperature=temperature,
                        mode="decision",
                        representation="image",
                        input_type="sentence",
                        sentence_id=sentence_id,
                        variant_id=variant_id,
                        prompt_id=decision_prompt_id,
                        prompt_hash=prompt_hash,
                        input_hash=input_hash,
                    )

                    if record.get_key() in existing_keys:
                        skipped_calls += 1
                        continue

                    total_calls += 1
                    if dry_run:
                        continue

                    try:
                        raw = _retry_call(
                            lambda: provider.infer_image(
                                provider_config.model_vision,
                                temperature,
                                image_path,
                                question,
                                DECISION_SYSTEM_PROMPT,
                            ),
                            max_retries=inference_cfg.max_retries,
                            sleep_s=inference_cfg.rate_limit_sleep,
                        )
                        parsed, _ = parse_yes_no(raw)
                        record.raw_response = raw
                        record.parsed_response = parsed
                    except Exception as exc:
                        record.parse_error = str(exc)
                        if inference_cfg.fail_fast:
                            raise

                    append_jsonl(jsonl_path, record)
                    existing_keys.add(record.get_key())

    # Similar processing for artifacts would go here...
    # (Following same pattern as sentences)

    if dry_run:
        logger.info(f"Dry run: would make {total_calls} calls, skipped {skipped_calls}")
    else:
        logger.info(f"Completed {total_calls} calls, skipped {skipped_calls} (already done)")

    return jsonl_path


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> pd.DataFrame:
    """Convert JSONL results to CSV for analysis."""
    records = []
    with jsonl_path.open("r") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return df
