"""IO helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(path: Path, rows: Iterable[dict], columns: List[str] | None = None) -> None:
    ensure_dir(path.parent)
    rows_list = list(rows)
    if not rows_list:
        if columns:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
        return
    fieldnames = columns or list(rows_list[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def append_csv(path: Path, rows: Iterable[dict], columns: List[str]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerows(list(rows))
