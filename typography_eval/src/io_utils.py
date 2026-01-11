from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from typography_eval import config
from typography_eval.src import fonts


@dataclass
class RunConfig:
    run_id: str
    model_text: str
    model_image: str
    temperature: float
    sentence_set_id: str
    variant_set_id: str
    notes: str = ""


def ensure_dirs() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path, mode: str = "w") -> None:
    if mode == "a" and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def generate_sentences_v1(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = config.DATA_DIR / "sentences_v1.csv"
    sentences = [
        (1, "The account status was updated earlier today", "neutral"),
        (2, "Your recent activity has been successfully recorded", "neutral"),
        (3, "This message contains information about your account", "neutral"),
        (4, "The system processed the request without errors", "neutral"),
        (5, "Your preferences were saved for future use", "neutral"),
        (6, "The document reflects the most recent changes", "neutral"),
        (7, "Please review the details before continuing", "cta"),
        (8, "You may want to confirm the information provided", "cta"),
        (9, "Consider updating your settings if needed", "cta"),
        (10, "Additional steps may be required to proceed", "cta"),
        (11, "You can review available options on the next screen", "cta"),
        (12, "Further action may improve your experience", "cta"),
        (13, "This notice applies to all registered users", "authority"),
        (14, "Access is granted based on current eligibility criteria", "authority"),
        (15, "The following requirements must be acknowledged", "authority"),
        (16, "Compliance with these guidelines is expected", "authority"),
        (17, "This policy governs acceptable use of the service", "authority"),
        (18, "The information below outlines required procedures", "authority"),
        (19, "Failure to complete this step may affect access", "warning"),
        (20, "Incomplete information could result in delays", "warning"),
        (21, "Certain actions may lead to unintended consequences", "warning"),
        (22, "Errors in submission can cause processing issues", "warning"),
        (23, "This action may impact your current settings", "warning"),
        (24, "Some features may not function as expected", "warning"),
        (25, "Discover features designed to improve your workflow", "promo"),
        (26, "This update introduces new capabilities for users", "promo"),
        (27, "Explore tools built to support your goals", "promo"),
        (28, "Enhanced options are now available for you", "promo"),
        (29, "Unlock additional benefits with updated settings", "promo"),
        (30, "New functionality is available in this release", "promo"),
        (31, "Enter the required information in the fields below", "procedural"),
        (32, "Follow the steps outlined to complete the process", "procedural"),
        (33, "Select an option to continue", "procedural"),
        (34, "Review the information before submitting the form", "procedural"),
        (35, "Use the menu to navigate available sections", "procedural"),
        (36, "Complete each section before proceeding", "procedural"),
    ]
    df = pd.DataFrame(
        sentences,
        columns=["sentence_id", "text", "sentence_category"],
    )
    df["sentence_set_id"] = config.SENTENCE_SET_ID
    df.to_csv(path, index=False)
    return df


def generate_variants_v1(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = config.DATA_DIR / "variants_v1.csv"
    df = pd.DataFrame(fonts.variants_table())
    df.to_csv(path, index=False)
    return df


def artifacts_from_metadata(metadata_path: Optional[Path] = None) -> pd.DataFrame:
    if metadata_path is None:
        metadata_path = config.METADATA_PATH
    metadata = pd.read_csv(metadata_path)
    artifacts = metadata.copy()
    if "representation" not in artifacts.columns:
        artifacts["representation"] = "image"
    if "variant_id" not in artifacts.columns:
        raise ValueError("metadata.csv must include variant_id")
    if "image_path" not in artifacts.columns:
        raise ValueError("metadata.csv must include image_path")
    artifacts["sentence_set_id"] = config.SENTENCE_SET_ID
    artifacts["variant_set_id"] = config.VARIANT_SET_ID
    artifacts["artifact_id"] = (
        artifacts["sentence_id"].astype(str)
        + ":"
        + artifacts["variant_id"].astype(str)
        + ":"
        + artifacts["representation"].astype(str)
    )
    cols = [
        "artifact_id",
        "sentence_id",
        "variant_id",
        "representation",
        "image_path",
        "sentence_set_id",
        "variant_set_id",
    ]
    artifacts = artifacts[cols]
    out_path = config.DATA_DIR / "artifacts_v1.csv"
    artifacts.to_csv(out_path, index=False)
    return artifacts


def ensure_results_schema(path: Path) -> None:
    if path.exists():
        return
    df = pd.DataFrame(
        columns=[
            "run_id",
            "sentence_id",
            "variant_id",
            "representation",
            "dimension",
            "model",
            "temperature",
            "response_raw",
            "response_norm",
            "response_01",
            "created_at",
        ]
    )
    df.to_csv(path, index=False)


def ensure_runs_schema(path: Path) -> None:
    if path.exists():
        return
    df = pd.DataFrame(
        columns=[
            "run_id",
            "created_at",
            "model_text",
            "model_image",
            "temperature",
            "sentence_set_id",
            "variant_set_id",
            "notes",
        ]
    )
    df.to_csv(path, index=False)


def next_run_id(runs_df: pd.DataFrame) -> str:
    if runs_df.empty or "run_id" not in runs_df.columns:
        return "run_000"
    existing = runs_df["run_id"].astype(str).tolist()
    indices = []
    for rid in existing:
        if rid.startswith("run_"):
            try:
                indices.append(int(rid.split("_")[-1]))
            except ValueError:
                continue
    next_idx = max(indices, default=-1) + 1
    return f"run_{next_idx:03d}"


def register_run(
    model_text: str = config.MODEL_TEXT,
    model_image: str = config.MODEL_IMAGE,
    temperature: float = config.TEMPERATURE,
    sentence_set_id: str = config.SENTENCE_SET_ID,
    variant_set_id: str = config.VARIANT_SET_ID,
    notes: str = "",
) -> RunConfig:
    ensure_dirs()
    ensure_runs_schema(config.RUNS_REGISTRY_PATH)

    runs_df = load_csv(config.RUNS_REGISTRY_PATH)
    run_id = next_run_id(runs_df)
    record = {
        "run_id": run_id,
        "created_at": utc_now(),
        "model_text": model_text,
        "model_image": model_image,
        "temperature": temperature,
        "sentence_set_id": sentence_set_id,
        "variant_set_id": variant_set_id,
        "notes": notes,
    }
    runs_df = pd.concat([runs_df, pd.DataFrame([record])], ignore_index=True)
    runs_df.to_csv(config.RUNS_REGISTRY_PATH, index=False)
    return RunConfig(**record)


def load_run(run_id: str) -> RunConfig:
    runs_df = load_csv(config.RUNS_REGISTRY_PATH)
    if runs_df.empty:
        raise ValueError("runs.csv is empty; register a run first")
    match = runs_df[runs_df["run_id"] == run_id]
    if match.empty:
        raise ValueError(f"run_id not found: {run_id}")
    row = match.iloc[0].to_dict()
    return RunConfig(**row)


def load_artifacts(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = config.DATA_DIR / "artifacts_v1.csv"
    df = pd.read_csv(path)
    return df


def load_sentences(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = config.DATA_DIR / "sentences_v1.csv"
    return pd.read_csv(path)


def load_variants(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = config.DATA_DIR / "variants_v1.csv"
    return pd.read_csv(path)


def add_artifacts_from_sentences_and_variants(
    sentences: pd.DataFrame,
    variants: pd.DataFrame,
    images_dir: Path,
) -> pd.DataFrame:
    rows = []
    for _, sentence in sentences.iterrows():
        sid = int(sentence["sentence_id"])
        for _, variant in variants.iterrows():
            variant_id = str(variant["variant_id"])
            image_path = images_dir / f"sentence_{sid:03d}" / f"{variant_id}.png"
            rows.append(
                {
                    "artifact_id": f"{sid}:{variant_id}:image",
                    "sentence_id": sid,
                    "variant_id": variant_id,
                    "representation": "image",
                    "image_path": str(image_path),
                    "sentence_set_id": config.SENTENCE_SET_ID,
                    "variant_set_id": config.VARIANT_SET_ID,
                }
            )
    return pd.DataFrame(rows)


def write_artifacts(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    if path is None:
        path = config.DATA_DIR / "artifacts_v1.csv"
    df.to_csv(path, index=False)
