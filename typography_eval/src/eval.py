from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from typography_eval import config
from typography_eval.src import io_utils
from typography_eval.src.metrics import parse_to_01


@dataclass
class EvalRecord:
    run_id: str
    sentence_id: int
    variant_id: str
    representation: str
    dimension: str
    model: str
    temperature: float
    response_raw: str
    response_norm: str
    response_01: Optional[int]
    created_at: str


def make_user_prompt_text(text: str, question: str) -> str:
    return f'Message:\n"{text}"\n\nQuestion:\n{question}'


def make_user_prompt_image(question: str) -> str:
    return f"Please evaluate the message shown in the image.\n\nQuestion:\n{question}"


def load_ocr_cache(ocr_dir: Optional[Path] = None) -> Dict[int, str]:
    if ocr_dir is None:
        ocr_dir = config.OCR_DIR
    cache: Dict[int, str] = {}
    for path in sorted(ocr_dir.glob("sentence_*.txt")):
        sid = int(path.stem.split("_")[-1])
        cache[sid] = path.read_text().strip()
    return cache


def load_existing_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def existing_keys(existing: pd.DataFrame) -> set:
    if existing.empty:
        return set()
    key_cols = [
        "run_id",
        "sentence_id",
        "variant_id",
        "representation",
        "dimension",
        "model",
        "temperature",
    ]
    return set(tuple(x) for x in existing[key_cols].astype(str).values.tolist())


def key_from_fields(
    run_id: str,
    sentence_id: int,
    variant_id: str,
    representation: str,
    dimension: str,
    model: str,
    temperature: float,
) -> tuple:
    return (
        str(run_id),
        str(sentence_id),
        str(variant_id),
        str(representation),
        str(dimension),
        str(model),
        str(temperature),
    )


def append_results(records: List[EvalRecord], path: Path, key_set: set) -> None:
    if not records:
        return
    df = pd.DataFrame([r.__dict__ for r in records])
    io_utils.write_csv(df, path, mode="a")
    for _, row in df.iterrows():
        key_set.add(
            (
                str(row["run_id"]),
                str(row["sentence_id"]),
                str(row["variant_id"]),
                str(row["representation"]),
                str(row["dimension"]),
                str(row["model"]),
                str(row["temperature"]),
            )
        )


def run_evaluation(
    run_id: str,
    artifacts: pd.DataFrame,
    ocr_cache: Dict[int, str],
    call_openai_text: Callable[[str, float, str, str], str],
    call_openai_image: Callable[[str, float, str, str], str],
    dimensions: Optional[Dict[str, str]] = None,
    model_text: str = config.MODEL_TEXT,
    model_image: str = config.MODEL_IMAGE,
    temperature: float = config.TEMPERATURE,
    flush_every: int = 50,
) -> None:
    io_utils.ensure_dirs()
    io_utils.ensure_results_schema(config.RESULTS_PATH)
    io_utils.load_run(run_id)

    if dimensions is None:
        dimensions = config.BINARY_QUESTIONS

    existing = load_existing_results(config.RESULTS_PATH)
    key_set = existing_keys(existing)

    buffer: List[EvalRecord] = []

    sentence_ids = sorted(artifacts["sentence_id"].unique())

    print("Starting OCR pass...")
    for sid in tqdm(sentence_ids):
        text = ocr_cache[int(sid)]
        for dim, question in dimensions.items():
            key = key_from_fields(
                run_id,
                int(sid),
                config.OCR_VARIANT_ID,
                "ocr",
                dim,
                model_text,
                temperature,
            )
            if key in key_set:
                continue
            raw = call_openai_text(model_text, temperature, text, question)
            parsed, norm = parse_to_01(raw)
            record = EvalRecord(
                run_id=run_id,
                sentence_id=int(sid),
                variant_id=config.OCR_VARIANT_ID,
                representation="ocr",
                dimension=dim,
                model=model_text,
                temperature=temperature,
                response_raw=raw,
                response_norm=norm,
                response_01=parsed,
                created_at=io_utils.utc_now(),
            )
            buffer.append(record)
            if len(buffer) >= flush_every:
                append_results(buffer, config.RESULTS_PATH, key_set)
                buffer = []

    append_results(buffer, config.RESULTS_PATH, key_set)
    buffer = []
    print("OCR pass complete.")

    print("Starting image pass...")
    for _, row in tqdm(artifacts.iterrows(), total=len(artifacts)):
        sid = int(row["sentence_id"])
        variant_id = str(row["variant_id"])
        image_path = row["image_path"]
        for dim, question in dimensions.items():
            key = key_from_fields(
                run_id,
                sid,
                variant_id,
                "image",
                dim,
                model_image,
                temperature,
            )
            if key in key_set:
                continue
            raw = call_openai_image(model_image, temperature, image_path, question)
            parsed, norm = parse_to_01(raw)
            record = EvalRecord(
                run_id=run_id,
                sentence_id=sid,
                variant_id=variant_id,
                representation="image",
                dimension=dim,
                model=model_image,
                temperature=temperature,
                response_raw=raw,
                response_norm=norm,
                response_01=parsed,
                created_at=io_utils.utc_now(),
            )
            buffer.append(record)
            if len(buffer) >= flush_every:
                append_results(buffer, config.RESULTS_PATH, key_set)
                buffer = []

    append_results(buffer, config.RESULTS_PATH, key_set)
    print("Image pass complete.")
