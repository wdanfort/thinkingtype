"""Execution engine for hrsim4.

Adapted from hrsim2/engine.py. Same JSONL-resume-on-restart, retry and
thread-pool shape; the differences are all consequences of v4's design:

  * one calibrated JD per resume (loop 2 tuned each occupation's JD-fit rank to
    a 40-60% accept rate), not two occupation-scoped draws;
  * renders are pre-built and QC'd, so the engine reads the manifest rather
    than rendering anything;
  * resumes are MULTI-PAGE documents (2-3 pages, OpenDyslexic 3.1), so image
    calls go through provider.infer_images, not infer_image;
  * no name x race interaction phase -- v2 found it null and uninterpretable at
    the floor, and v4 spends the budget on the ink ladder instead.
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

from typo_eval.hrsim4.config import HrsimV4Config, TEXT_ARM
from typo_eval.hrsim4.prompts import (
    PROMPT_ID,
    SYSTEM_PROMPT,
    make_question,
    parse_decision,
)
from typo_eval.hrsim.config import HrsimLegConfig
from typo_eval.prompts import compute_prompt_hash
from typo_eval.providers import get_provider

logger = logging.getLogger(__name__)


@dataclass
class Hrsim4Record:
    """Canonical record schema for hrsim4 JSONL logging."""

    run_tag: str
    timestamp: str
    leg: str
    provider: str
    model: str
    arm: str  # "text" | "image"
    variant_id: str  # "TEXT" for the text arm, else the render arm id
    item_id: str
    resume_id: int
    category: str
    jd_id: int
    jd_job: str
    jd_rank: int
    boundary_sensitive: bool
    pages: Optional[int]
    prompt_id: str
    prompt_hash: str
    input_hash: str
    repeat_idx: int
    request_seed: Optional[int] = None
    raw_response: str = ""
    parsed_decision: Optional[int] = None
    parse_error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_key(self) -> Tuple:
        return (self.item_id, self.arm, self.variant_id, self.repeat_idx)


def load_existing_keys(jsonl_path: Path) -> Set[Tuple]:
    """Keys already logged. Transport failures are NOT counted as done, so a
    resumed run retries them (the v1 lesson: logged parse_error records were
    treated as complete and 429-poisoned cells never got re-run)."""
    keys: Set[Tuple] = set()
    if not jsonl_path.exists():
        return keys
    with jsonl_path.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            parse_error = rec.get("parse_error")
            if parse_error and str(parse_error).startswith("exception"):
                continue
            try:
                keys.add((rec["item_id"], rec["arm"], rec["variant_id"], rec["repeat_idx"]))
            except KeyError:
                continue
    return keys


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
            logger.warning(f"Retry {attempt + 1}/{max_retries}: {exc}, sleeping {delay:.1f}s")
            time.sleep(delay)
    return None


# ---------------------------------------------------------------------------
# Item set
# ---------------------------------------------------------------------------

def build_items(config: HrsimV4Config, repo_root: Path) -> pd.DataFrame:
    """Assemble the calibrated item set: one JD per resume at its occupation's
    calibrated rank, joined to the render manifest and the gate-3 sensitivity
    flags."""
    corpus = repo_root / config.corpus_dir
    caldir = repo_root / config.calibration_dir

    # Calibration is model-specific: config.ranks_path points a leg at its own
    # calibrated boundary. Defaults to the run root's (the Sonnet) calibration.
    ranks_file = (repo_root / config.ranks_path if config.ranks_path
                  else caldir / "calibrated_ranks.csv")
    if not ranks_file.exists():
        raise FileNotFoundError(f"no calibrated ranks at {ranks_file}")
    ranks = pd.read_csv(ranks_file).set_index("category")["rank"]

    cand = pd.read_csv(corpus / "candidates.csv")
    if "jd_text" not in cand.columns:
        # candidates.csv stores jd_id only; posting text lives in the sidecar so
        # the ladder can go deep without duplicating 1.2KB of text per row.
        jd = pd.read_csv(corpus / "jd_lookup.csv").set_index("jd_id")["jd_text"]
        cand["jd_text"] = cand.jd_id.map(jd)
    manifest = pd.read_csv(corpus / "render_manifest.csv")

    sens_path = caldir / "sensitive_items.csv"
    sensitive = set()
    if sens_path.exists():
        s = pd.read_csv(sens_path)
        sensitive = {(r.category, int(r.resume_id)) for r in s.itertuples()}

    rows = []
    for occ, rank in ranks.items():
        sub = cand[(cand.category == occ) & (cand["rank"] == int(rank))]
        for _, r in sub.iterrows():
            rows.append({
                "item_id": f"{int(r.resume_id)}_{int(r.jd_id)}",
                "resume_id": int(r.resume_id),
                "category": occ,
                "jd_id": int(r.jd_id),
                "jd_job": r.jd_job,
                "jd_text": r.jd_text,
                "jd_rank": int(rank),
                "boundary_sensitive": (occ, int(r.resume_id)) in sensitive,
            })
    items = pd.DataFrame(rows)

    if config.items == "sensitive":
        items = items[items.boundary_sensitive].reset_index(drop=True)
    elif config.items != "all":
        raise ValueError(f"config.items must be 'all' or 'sensitive', got {config.items!r}")

    # Resume plain text for the text arm.
    texts, pages = {}, {}
    for p in (corpus / "resumes").glob("*.json"):
        d = json.loads(p.read_text())
        texts[int(d["resume_id"])] = d["plain_text"]
    items["resume_text"] = items.resume_id.map(texts)
    missing = items[items.resume_text.isna()]
    if len(missing):
        raise ValueError(f"{len(missing)} items have no resume text: "
                         f"{missing.resume_id.tolist()[:5]}")

    # Render paths, pipe-separated page lists from the manifest.
    renders = {}
    for _, m in manifest.iterrows():
        renders[(int(m.resume_id), m.arm)] = (
            str(m.image_paths).split("|"), int(m.pages)
        )
    items.attrs["renders"] = renders
    return items


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class _Runner:
    def __init__(self, config: HrsimV4Config, leg: HrsimLegConfig, jsonl_path: Path):
        self.config = config
        self.leg = leg
        self.jsonl_path = jsonl_path
        self._local = threading.local()
        self._write_lock = threading.Lock()

    def _provider(self):
        if not hasattr(self._local, "provider"):
            p = get_provider(self.leg.provider)
            tier = self.config.inference.openai_service_tier
            if self.leg.provider == "openai" and tier and hasattr(p, "service_tier"):
                p.service_tier = tier
            self._local.provider = p
        return self._local.provider

    def _append(self, record: Hrsim4Record) -> None:
        with self._write_lock:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
                f.flush()

    def _execute(self, record: Hrsim4Record, payload: Dict[str, Any]) -> None:
        inf = self.config.inference
        provider = self._provider()
        seed = self.config.seed * 1000 + record.repeat_idx
        try:
            if record.arm == "text":
                raw = _retry_call(
                    lambda: provider.infer_text(
                        self.leg.model, inf.temperature, payload["text"],
                        payload["question"], SYSTEM_PROMPT, seed=seed,
                    ),
                    max_retries=inf.max_retries, sleep_s=inf.rate_limit_sleep,
                )
            else:
                raw = _retry_call(
                    lambda: provider.infer_images(
                        self.leg.model, inf.temperature, payload["image_paths"],
                        payload["question"], SYSTEM_PROMPT, seed=seed,
                    ),
                    max_retries=inf.max_retries, sleep_s=inf.rate_limit_sleep,
                )
            parsed, parse_error = parse_decision(raw)
            record.raw_response = raw or ""
            record.parsed_decision = parsed
            record.parse_error = parse_error
            record.request_seed = seed
            usage = getattr(provider, "last_usage", None) or {}
            record.input_tokens = usage.get("input_tokens")
            record.output_tokens = usage.get("output_tokens")
        except Exception as exc:
            record.parse_error = f"exception: {exc}"
        self._append(record)

    def run(self, tasks: List[Tuple[Hrsim4Record, Dict[str, Any]]], desc: str) -> None:
        workers = max(1, self.config.inference.concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._execute, r, p) for r, p in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=desc):
                pass


def _make_tasks(
    config: HrsimV4Config,
    leg: HrsimLegConfig,
    items: pd.DataFrame,
    existing_keys: Set[Tuple],
) -> List[Tuple[Hrsim4Record, Dict[str, Any]]]:
    renders = items.attrs["renders"]
    tasks: List[Tuple[Hrsim4Record, Dict[str, Any]]] = []

    for _, row in items.iterrows():
        question = make_question(row["jd_text"])
        prompt_hash = compute_prompt_hash(SYSTEM_PROMPT, question)
        resume_id = int(row["resume_id"])

        for variant_id in config.arms():
            is_text = variant_id == TEXT_ARM
            if is_text:
                payload = {"text": row["resume_text"], "question": question}
                input_hash = _compute_input_hash(row["resume_text"])
                pages = None
            else:
                entry = renders.get((resume_id, variant_id))
                if not entry:
                    logger.warning(f"No render for resume {resume_id} / {variant_id}")
                    continue
                paths, pages = entry
                missing = [p for p in paths if not Path(p).exists()]
                if missing:
                    logger.warning(f"Missing page images for {resume_id}/{variant_id}: {missing[:2]}")
                    continue
                payload = {"image_paths": paths, "question": question}
                # Hash all pages so a re-render is detectable, not just page 1.
                input_hash = _compute_input_hash(
                    b"".join(Path(p).read_bytes() for p in paths)
                )

            for repeat_idx in range(config.repeats):
                record = Hrsim4Record(
                    run_tag=config.run_tag,
                    timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                    leg=leg.name, provider=leg.provider, model=leg.model,
                    arm="text" if is_text else "image",
                    variant_id=variant_id,
                    item_id=row["item_id"], resume_id=resume_id,
                    category=row["category"], jd_id=int(row["jd_id"]),
                    jd_job=row["jd_job"], jd_rank=int(row["jd_rank"]),
                    boundary_sensitive=bool(row["boundary_sensitive"]),
                    pages=pages,
                    prompt_id=PROMPT_ID, prompt_hash=prompt_hash,
                    input_hash=input_hash, repeat_idx=repeat_idx,
                )
                if record.get_key() in existing_keys:
                    continue
                tasks.append((record, payload))
    return tasks


def run_leg(
    config: HrsimV4Config,
    leg: HrsimLegConfig,
    items: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Run one model leg to completion, resuming from whatever is on disk."""
    jsonl_path = out_dir / f"raw_{leg.name}.jsonl"
    existing = load_existing_keys(jsonl_path)
    tasks = _make_tasks(config, leg, items, existing)

    planned = len(items) * len(config.arms()) * config.repeats
    logger.info(
        f"leg {leg.name}: {planned} cells planned, {planned - len(tasks)} already "
        f"logged, {len(tasks)} to run"
    )
    if tasks:
        _Runner(config, leg, jsonl_path).run(tasks, desc=f"hrsim4 {leg.name}")
    return load_records(jsonl_path)
