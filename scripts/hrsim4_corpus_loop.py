"""Loop 1: rebuild the hrsim4 resume corpus until every QC gate passes.

Runs build -> audit -> report, and repeats with a tightened admission filter
whenever a gate fails. Zero API cost: everything here is parsing, layout and
image measurement. The loop stops only when `qc.audit` returns no failures.

    python scripts/hrsim4_corpus_loop.py --per-occupation 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from typo_eval.hrsim4 import qc  # noqa: E402
from typo_eval.hrsim4.build import ARMS, OCCUPATIONS, build_corpus  # noqa: E402

RAW = REPO / "data" / "inputs" / "resumes_raw" / "Resume.csv"
FONTS = REPO / "assets" / "fonts"
OUT = REPO / "data" / "inputs" / "hrsim4"


def run(per_occupation: int, max_revs: int, render: bool) -> int:
    font_paths = {a.id: FONTS / a.font_file for a in ARMS}
    history = []

    for rev in range(1, max_revs + 1):
        print(f"\n{'=' * 70}\nREV {rev}: building {per_occupation}/occupation "
              f"x {len(ARMS)} arms\n{'=' * 70}")
        manifest = build_corpus(
            RAW, per_occupation=per_occupation, fonts_dir=FONTS,
            out_dir=OUT, render=render,
        )
        gates = qc.audit(
            manifest, corpus_dir=OUT, expected_arms=len(ARMS),
            per_occupation=per_occupation, measure_images=render,
            font_paths=font_paths,
        )
        print(qc.report(gates))
        failing = [g for g in gates if not g.ok]
        history.append({"rev": rev, "failing": [g.name for g in failing]})
        (OUT / "qc_history.json").write_text(json.dumps(history, indent=1))
        (OUT / "qc_report.txt").write_text(qc.report(gates))

        if not failing:
            print(f"\nALL GATES PASS at rev {rev}.")
            rej = pd.read_csv(OUT / "rejected_resumes.csv")
            print(f"corpus: {manifest.resume_id.nunique()} resumes, "
                  f"{len(manifest)} resume x arm records, "
                  f"{int(manifest.pages.sum())} page images")
            print(f"rejected during sampling: {len(rej)}")
            return 0

        print(f"\nREV {rev} FAILED: {[g.name for g in failing]}")
        # Gates that admission-filtering can fix are handled by tightening
        # thresholds; anything else is a code defect and must stop the loop
        # rather than be silently retried.
        fixable = {"page_count_sane", "reference_arm_is_1_2_pages",
                   "no_truncation", "no_residual_defects"}
        if not set(g.name for g in failing) <= fixable:
            print("Unfixable by filtering (code defect) -- stopping.")
            return 1
        qc.MAX_CHARS = int(qc.MAX_CHARS * 0.92)
        print(f"tightening qc.MAX_CHARS -> {qc.MAX_CHARS} and rebuilding")

    print(f"\nExhausted {max_revs} revisions without passing.")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-occupation", type=int, default=50)
    ap.add_argument("--max-revs", type=int, default=6)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    raise SystemExit(run(a.per_occupation, a.max_revs, render=not a.no_render))
