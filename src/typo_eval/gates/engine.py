"""Calibration and vision inference for the gates experiment.

Two phases share one machinery:
- calibration: text-mode, k repeats per item, over the full stimulus bank.
  Yields p_yes per item per provider, used to select boundary-adjacent items.
- vision: image-mode, k repeats per selected item per visual variant.

Both phases log to JSONL with resume support (same pattern as the v0
harness) and run calls on a thread pool with per-thread provider clients.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from typo_eval.gates.config import GatesConfig
from typo_eval.gates.prompts import (
    get_gate_spec,
    make_gate_image_prompt,
    make_gate_text_prompt,
)
from typo_eval.prompts import compute_prompt_hash, parse_yes_no
from typo_eval.providers import get_provider

logger = logging.getLogger(__name__)


@dataclass
class GateRecord:
    """Canonical record schema for gate JSONL logging."""

    run_tag: str
    timestamp: str
    provider: str
    model: str
    phase: str  # "calibration" | "vision"
    representation: str  # "text" | "image"
    gate: str
    item_id: str
    scenario: str
    level: int
    variant_id: str  # "__text__" for the text arm
    repeat_idx: int
    prompt_id: str
    prompt_hash: str
    input_hash: str
    raw_response: str = ""
    parsed_response: Optional[int] = None
    parse_error: Optional[str] = None
    request_seed: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_key(self) -> Tuple:
        return (
            self.phase,
            self.representation,
            self.gate,
            self.item_id,
            self.variant_id,
            self.repeat_idx,
            self.provider,
            self.model,
            self.prompt_hash,
        )


def load_existing_keys(jsonl_path: Path) -> Set[Tuple]:
    keys: Set[Tuple] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open("r") as f:
        for line in f:
            try:
                record = GateRecord(**json.loads(line.strip()))
                keys.add(record.get_key())
            except (json.JSONDecodeError, TypeError):
                continue
    return keys


def _compute_input_hash(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode()
    return hashlib.sha256(content).hexdigest()[:16]


def _retry_call(fn, max_retries: int, sleep_s: float, jitter: float = 0.25):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise exc
            delay = sleep_s * (2**attempt) * (1 + random.uniform(-jitter, jitter))
            logger.warning(
                f"Retry {attempt + 1}/{max_retries}: {exc}, sleeping {delay:.1f}s"
            )
            time.sleep(delay)
    return None


class _Runner:
    """Executes a batch of pending GateRecords on a thread pool."""

    def __init__(self, config: GatesConfig, provider_name: str, jsonl_path: Path):
        self.config = config
        self.provider_name = provider_name
        self.provider_config = getattr(config.providers, provider_name)
        self.jsonl_path = jsonl_path
        self._local = threading.local()
        self._write_lock = threading.Lock()

    def _provider(self):
        if not hasattr(self._local, "provider"):
            p = get_provider(self.provider_name)
            p.service_tier = getattr(self.provider_config, "service_tier", None)
            p.thinking_budget = getattr(self.provider_config, "thinking_budget", None)
            self._local.provider = p
        return self._local.provider

    def _append(self, record: GateRecord) -> None:
        with self._write_lock:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
                f.flush()

    def _execute(self, record: GateRecord, payload: Dict[str, Any]) -> None:
        inf = self.config.inference
        provider = self._provider()
        # Distinct seed per repeat so OpenAI repeats sample independently
        # (providers without seed support ignore it).
        seed = inf.seed * 1000 + record.repeat_idx
        try:
            if record.representation == "text":
                raw = _retry_call(
                    lambda: provider.infer_text(
                        payload["model"],
                        inf.temperature,
                        payload["text"],
                        payload["question"],
                        payload["system_prompt"],
                        seed=seed,
                    ),
                    max_retries=inf.max_retries,
                    sleep_s=inf.rate_limit_sleep,
                )
            else:
                raw = _retry_call(
                    lambda: provider.infer_image(
                        payload["model"],
                        inf.temperature,
                        payload["image_path"],
                        payload["question"],
                        payload["system_prompt"],
                        seed=seed,
                    ),
                    max_retries=inf.max_retries,
                    sleep_s=inf.rate_limit_sleep,
                )
            parsed, _ = parse_yes_no(raw)
            record.raw_response = raw or ""
            record.parsed_response = parsed
            record.request_seed = seed
            usage = getattr(provider, "last_usage", None) or {}
            record.input_tokens = usage.get("input_tokens")
            record.output_tokens = usage.get("output_tokens")
        except Exception as exc:
            record.parse_error = str(exc)
        self._append(record)

    def run(self, tasks: List[Tuple[GateRecord, Dict[str, Any]]], desc: str) -> None:
        workers = max(1, self.config.inference.workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._execute, record, payload)
                for record, payload in tasks
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=desc):
                pass


def _make_text_tasks(
    config: GatesConfig,
    provider_name: str,
    items_df: pd.DataFrame,
    repeats: int,
    phase: str,
    existing_keys: Set[Tuple],
) -> List[Tuple[GateRecord, Dict[str, Any]]]:
    provider_config = getattr(config.providers, provider_name)
    variant = config.inference.prompt_variant
    tasks = []
    for _, row in items_df.iterrows():
        spec = get_gate_spec(row["gate"], variant)
        user_prompt = make_gate_text_prompt(row["text"], spec.question)
        prompt_hash = compute_prompt_hash(spec.system_prompt, user_prompt)
        for repeat_idx in range(repeats):
            record = GateRecord(
                run_tag=config.run_tag,
                timestamp=dt.datetime.utcnow().isoformat(),
                provider=provider_name,
                model=provider_config.model_text,
                phase=phase,
                representation="text",
                gate=row["gate"],
                item_id=row["item_id"],
                scenario=row["scenario"],
                level=int(row["level"]),
                variant_id="__text__",
                repeat_idx=repeat_idx,
                prompt_id=spec.prompt_id,
                prompt_hash=prompt_hash,
                input_hash=_compute_input_hash(row["text"]),
            )
            if record.get_key() in existing_keys:
                continue
            payload = {
                "model": provider_config.model_text,
                "text": row["text"],
                "question": spec.question,
                "system_prompt": spec.system_prompt,
            }
            tasks.append((record, payload))
    return tasks


def _make_image_tasks(
    config: GatesConfig,
    provider_name: str,
    metadata_df: pd.DataFrame,
    repeats: int,
    existing_keys: Set[Tuple],
) -> List[Tuple[GateRecord, Dict[str, Any]]]:
    provider_config = getattr(config.providers, provider_name)
    variant = config.inference.prompt_variant
    tasks = []
    for _, row in metadata_df.iterrows():
        spec = get_gate_spec(row["gate"], variant)
        user_prompt = make_gate_image_prompt(spec.question)
        prompt_hash = compute_prompt_hash(spec.system_prompt, user_prompt)
        image_path = Path(row["image_path"])
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        input_hash = _compute_input_hash(image_path.read_bytes())
        for repeat_idx in range(repeats):
            record = GateRecord(
                run_tag=config.run_tag,
                timestamp=dt.datetime.utcnow().isoformat(),
                provider=provider_name,
                model=provider_config.model_vision,
                phase="vision",
                representation="image",
                gate=row["gate"],
                item_id=row["item_id"],
                scenario=row["scenario"],
                level=int(row["level"]),
                variant_id=row["variant_id"],
                repeat_idx=repeat_idx,
                prompt_id=spec.prompt_id,
                prompt_hash=prompt_hash,
                input_hash=input_hash,
            )
            if record.get_key() in existing_keys:
                continue
            payload = {
                "model": provider_config.model_vision,
                "image_path": str(image_path),
                "question": spec.question,
                "system_prompt": spec.system_prompt,
            }
            tasks.append((record, payload))
    return tasks


def load_records(jsonl_path: Path) -> pd.DataFrame:
    records = []
    if jsonl_path.exists():
        with jsonl_path.open("r") as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(records)


def run_calibration(
    config: GatesConfig,
    provider_name: str,
    items_df: pd.DataFrame,
    run_dir: Path,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Path:
    """Text-mode repeated sampling over the full stimulus bank."""
    jsonl_path = run_dir / "calibration" / f"{provider_name}.jsonl"
    existing = load_existing_keys(jsonl_path)
    tasks = _make_text_tasks(
        config, provider_name, items_df, config.calibration.repeats,
        "calibration", existing,
    )
    if limit:
        tasks = tasks[:limit]
    logger.info(
        f"Calibration [{provider_name}]: {len(tasks)} pending calls "
        f"({len(existing)} already logged)"
    )
    if dry_run or not tasks:
        return jsonl_path
    _Runner(config, provider_name, jsonl_path).run(
        tasks, desc=f"Calibrating ({provider_name})"
    )
    return jsonl_path


def summarize_calibration(jsonl_path: Path) -> pd.DataFrame:
    """Aggregate calibration JSONL into per-item p_yes."""
    df = load_records(jsonl_path)
    if df.empty:
        return pd.DataFrame()
    ok = df[df["parsed_response"].notna()]
    grouped = (
        ok.groupby(["provider", "model", "gate", "item_id", "scenario", "level"])
        .agg(
            n=("parsed_response", "size"),
            n_yes=("parsed_response", "sum"),
        )
        .reset_index()
    )
    grouped["p_yes"] = grouped["n_yes"] / grouped["n"]
    n_err = int(df["parsed_response"].isna().sum())
    if n_err:
        logger.warning(f"{n_err} calibration calls failed to parse")
    return grouped


def _flip_distances(grp: pd.DataFrame) -> pd.Series:
    """Distance from each item's ladder level to its scenario's flip point.

    Models are nearly deterministic at default sampling, so most items have
    p_yes of exactly 0 or 1 and the decision boundary shows up as the level
    where a scenario's majority answer flips. Items adjacent to that flip
    are boundary cases even when their own p_yes is extreme. Scenarios with
    no flip (all yes or all no) get inf.
    """
    dist = pd.Series(float("inf"), index=grp.index)
    for _, scen in grp.groupby("scenario"):
        scen = scen.sort_values("level")
        levels = scen["level"].to_numpy()
        majority = (scen["p_yes"] > 0.5).to_numpy()
        boundaries = [
            (levels[i] + levels[i + 1]) / 2
            for i in range(len(levels) - 1)
            if majority[i] != majority[i + 1]
        ]
        if boundaries:
            for idx, level in zip(scen.index, levels):
                dist[idx] = min(abs(level - b) for b in boundaries)
    return dist


def select_boundary_items(
    calibration_df: pd.DataFrame, config: GatesConfig
) -> pd.DataFrame:
    """Pick the n_select boundary-adjacent items per gate.

    Priority: (1) items whose repeated text answers split (p_yes within the
    band) — the model is measurably torn; (2) items nearest their scenario's
    flip point on the ladder; (3) |p_yes - 0.5|. The achieved boundary
    quality is reported by the analysis either way.
    """
    lo, hi = config.calibration.band
    n_select = config.calibration.n_select
    picks = []
    for gate, grp in calibration_df.groupby("gate"):
        grp = grp.copy()
        grp["dist"] = (grp["p_yes"] - 0.5).abs()
        grp["in_band"] = (grp["p_yes"] >= lo) & (grp["p_yes"] <= hi)
        grp["flip_dist"] = _flip_distances(grp)
        grp = grp.sort_values(
            ["in_band", "flip_dist", "dist", "item_id"],
            ascending=[False, True, True, True],
        )
        picks.append(grp.head(n_select))
    if not picks:
        return pd.DataFrame()
    return pd.concat(picks, ignore_index=True).drop(columns=["dist"])


def run_vision(
    config: GatesConfig,
    provider_name: str,
    selected_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    run_dir: Path,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Path:
    """Image-mode repeated sampling over selected items x variants."""
    jsonl_path = run_dir / "vision" / f"{provider_name}.jsonl"
    selected_ids = set(selected_df["item_id"])
    variant_ids = {v.id for v in config.variants}
    meta = metadata_df[
        metadata_df["item_id"].isin(selected_ids)
        & metadata_df["variant_id"].isin(variant_ids)
    ]
    existing = load_existing_keys(jsonl_path)
    tasks = _make_image_tasks(
        config, provider_name, meta, config.vision.repeats, existing
    )
    if limit:
        tasks = tasks[:limit]
    logger.info(
        f"Vision [{provider_name}]: {len(tasks)} pending calls "
        f"({len(existing)} already logged)"
    )
    if dry_run or not tasks:
        return jsonl_path
    _Runner(config, provider_name, jsonl_path).run(
        tasks, desc=f"Vision ({provider_name})"
    )
    return jsonl_path
