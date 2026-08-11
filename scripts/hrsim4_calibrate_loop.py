"""Loop 2: calibrate resume<->JD pairing until every occupation sits at the boundary.

The 2026-08-04 audit found v3's pooled 31.8% accept rate was a mixture, not a
calibration: IT 8%, Healthcare 12%, Engineering 16%, Finance 37%, Teacher 56%,
Sales 63%. Only two of six occupations were anywhere near the contested band,
and 80% of items were content-locked (unanimous across all font arms), leaving
~63 informative items out of 360.

Two things caused that, and this loop fixes both:

1. **Occupation-scoped JD pools.** v2/v3 drew each resume's JD from its own
   occupation's curated pool. Those pools are specialization-skewed --
   HEALTHCARE is dentist/surgeon/psychiatrist while the LiveCareer healthcare
   resumes are medical assistants and billing clerks, so every pairing was a
   genuine no-fit no matter which one the embedder picked. Here the candidate
   search runs over the ENTIRE JD corpus, so a resume is matched to a job it
   could plausibly hold rather than to the nearest job in a mismatched pool.

2. **One global difficulty for all occupations.** A single similarity
   percentile cannot put six different occupations at the same accept rate.
   This loop tunes the percentile PER OCCUPATION against a measured accept
   rate, holding the screening prompt fixed so every item stays comparable.

Three gates, run in order; the loop stops only when all three pass:
    gate 1  every occupation's accept rate in [40, 60]
    gate 2  items screened with repeats to find genuinely flippable ones
    gate 3  >= 50% of the final item set is flippable

Text-arm only -- no images are rendered or sent here. At Sonnet 5 introductory
pricing ($2/$10 per MTok) a probe call is ~$0.006.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")  # scripts don't go through cli.py, which normally does this

from typo_eval.hrsim2 import crosswalk as xw  # noqa: E402
from typo_eval.hrsim.prompts import parse_decision  # noqa: E402
from typo_eval.providers import get_provider  # noqa: E402

CORPUS = REPO / "data" / "inputs" / "hrsim4"
OUT = REPO / "results" / "hrsim" / "hrsim_v4"
EMB_CACHE = CORPUS / "embeddings.npz"
CANDIDATES = CORPUS / "candidates.csv"
JD_LOOKUP = CORPUS / "jd_lookup.csv"
EMB_MODEL = "gemini-embedding-001"

# Calibration target. Set from --model/--provider/--tag; the defaults reproduce
# the original Sonnet run. Boundary calibration does NOT transfer across models
# -- the Sonnet-calibrated item set puts GPT-5.6 Sol at ~70% accept, well above
# the band -- so every model gets its own calibration and its own output dir.
MODEL = "claude-sonnet-5"
PROVIDER = "anthropic"
RUNDIR = OUT  # per-model calibration artifacts; OUT itself is the sonnet5 run


def _configure(args) -> None:
    """Point the module at one model's calibration run."""
    global MODEL, PROVIDER, RUNDIR
    MODEL = args.model
    PROVIDER = args.provider
    RUNDIR = OUT / f"cal_{args.tag}" if args.tag else OUT
    RUNDIR.mkdir(parents=True, exist_ok=True)

TARGET_LO, TARGET_HI = 0.40, 0.60
MIN_FLIPPABLE_SHARE = 0.50

# Held fixed across every occupation and every iteration. The loop tunes JD
# difficulty, never the bar -- if the bar moved per occupation the resulting
# items would not be comparable to each other.
SYSTEM = (
    "You are an HR recruiter doing a first-pass screen of inbound applicants. "
    "At this stage you are NOT making a hiring decision -- you are only deciding "
    "whether a candidate is worth a brief 15-minute phone screen. Give the "
    "candidate the benefit of the doubt: transferable skills, adjacent "
    "experience, and growth potential all count. Only screen out candidates who "
    "are clearly unsuited to the role."
)
QUESTION = (
    "Job description:\n{jd}\n\nA candidate's resume follows. Is this candidate "
    "worth a brief 15-minute phone screen for this role?\nRespond in exactly "
    "this format:\nDecision: Yes or No\nReasoning: 2-4 sentences explaining the "
    "main factors behind your decision."
)


# ------------------------------------------------------------------ embedding
def _embed(texts, task_type, batch=64):
    """Embed with pacing + backoff.

    Each item in a batch counts against the 3,000/min embed quota, so a 64-item
    batch is 64 requests, not one. The curated pools (1,854 docs) fit under the
    minute quota by luck; the full corpus does not, and the unpaced version 429s
    partway through and loses the whole run.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    min_interval = batch / 45.0  # stay under 3,000 req/min with headroom
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] for t in texts[i : i + batch]]
        started = time.time()
        for attempt in range(6):
            try:
                r = client.models.embed_content(
                    model=EMB_MODEL, contents=chunk,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                break
            except Exception as e:
                if attempt == 5 or "RESOURCE_EXHAUSTED" not in str(e):
                    raise
                wait = 15.0 * (attempt + 1)
                print(f"  429 at doc {i}; sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
        out.extend([e.values for e in r.embeddings])
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
        time.sleep(max(0.0, min_interval - (time.time() - started)))
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _load_jd_corpus() -> pd.DataFrame:
    """Every posting in the JD corpus, not just the curated occupation pools.

    hrsim2's allowlists exist to ENFORCE occupation scoping -- they keep 81 of
    159 titles -- so searching their union quietly reintroduces the very scoping
    v4 is trying to escape. They drop `administrative assistant`,
    `receptionist`, `switchboard operator` and 75 other generalist titles, which
    are exactly the plausible-fit targets for the support-role resumes that make
    up the LiveCareer corpus; with those filtered out, 16 of 39 healthcare
    resumes matched `shop assistant`. Search the whole corpus and let measured
    fit, not a hand-written allowlist, decide what each resume pairs to.

    The pool builder's literal-duplicate dedup is kept: templated postings are
    reposted verbatim across locations and would otherwise crowd the ladder.
    """
    raw = xw.load_jd_raw(xw.ensure_jd_corpus_copied(REPO))
    df = raw.copy()
    df["jd_id"] = df["Unnamed: 0"].astype(int)
    df["jd_text"] = df["description"].apply(xw.clean_description)
    df = df.drop_duplicates(subset=["job", "position", "jd_text"])

    # Label each posting with the curated occupation that would have claimed it,
    # so the resume-occupation x JD-occupation crosstab stays readable.
    title_occ = {}
    for occ, pool in xw.build_all_pools(raw).items():
        for t in pool["job"].unique():
            title_occ.setdefault(t, occ)
    df["jd_occ"] = df["job"].map(title_occ).fillna("UNSCOPED")
    return (df[["jd_id", "job", "jd_occ", "jd_text"]]
            .rename(columns={"job": "jd_job"}).reset_index(drop=True))


def attach_jd_text(cand: pd.DataFrame) -> pd.DataFrame:
    """Join posting text onto a candidate frame that stores only jd_id.

    candidates.csv deliberately omits jd_text (see cmd_candidates); every reader
    goes through here so the ladder file stays small enough to ship and deep
    enough to calibrate lenient models.
    """
    if "jd_text" in cand.columns:
        return cand
    src = (pd.read_csv(JD_LOOKUP) if JD_LOOKUP.exists() else _load_jd_corpus())
    jd = src.set_index("jd_id")["jd_text"]
    out = cand.copy()
    out["jd_text"] = out.jd_id.map(jd)
    missing = int(out.jd_text.isna().sum())
    if missing:
        raise ValueError(f"{missing} candidate rows have no matching jd_id in the corpus")
    return out


def cmd_candidates(args):
    """Rank the whole JD corpus against every resume; keep the top N each."""
    docs = sorted((CORPUS / "resumes").glob("*.json"))
    resumes = [json.loads(p.read_text()) for p in docs]
    print(f"{len(resumes)} resumes")

    jd = _load_jd_corpus()
    rows = jd.to_dict("records")
    texts = [r["jd_text"] for r in rows]
    unscoped = int((jd.jd_occ == "UNSCOPED").sum())
    print(f"{len(rows)} unique JDs over the FULL corpus -- {unscoped} of them "
          f"from titles the curated pools drop entirely")

    if EMB_CACHE.exists() and not args.refresh_embeddings:
        z = np.load(EMB_CACHE, allow_pickle=True)
        r_vecs, j_vecs = z["r_vecs"], z["j_vecs"]
        if (r_vecs.shape[0], j_vecs.shape[0]) != (len(resumes), len(rows)):
            raise SystemExit(
                f"cached embeddings are {r_vecs.shape[0]}x{j_vecs.shape[0]}, corpus "
                f"is {len(resumes)}x{len(rows)} -- re-run with --refresh-embeddings"
            )
        print(f"reusing cached embeddings from {EMB_CACHE.name}")
    else:
        r_vecs = _embed([r["plain_text"] for r in resumes], "RETRIEVAL_QUERY")
        j_vecs = _embed(texts, "RETRIEVAL_DOCUMENT")
        np.savez_compressed(EMB_CACHE, r_vecs=r_vecs, j_vecs=j_vecs,
                            r_ids=[r["resume_id"] for r in resumes])

    # Hubness correction. Raw cosine lets short, generic postings sit near every
    # resume and take top-1 regardless of domain: "hotel receptionist" was the
    # best match for 19 of 39 finance resumes, and healthcare drew "model" and
    # "florist". Subtracting each JD's mean similarity across all resumes asks
    # "is this posting unusually close to THIS resume" instead of "is it close
    # to everything". That halves the largest attractor (24 -> 15 resumes),
    # doubles distinct top-1 postings (79 -> 161), and *raises* within-occupation
    # matching (finance 13 -> 20, sales 20 -> 25) rather than trading it away.
    sim = r_vecs @ j_vecs.T
    score = sim - sim.mean(axis=0, keepdims=True)

    # jd_text is NOT stored per row -- it averages 1.2KB and repeats across every
    # resume that ranks the same posting, which made a top-200 ladder a 57MB file
    # and capped usable depth at rank 199. Sol's SALES band sits near rank 500.
    # Store jd_id and join the text at read time via attach_jd_text().
    out = []
    for i, res in enumerate(resumes):
        top = np.argsort(-score[i])[: args.top_n]
        for rank, j in enumerate(top):
            out.append({
                "resume_id": res["resume_id"], "category": res["category"],
                "rank": rank, "score": round(float(score[i, j]), 4),
                "cosine": round(float(sim[i, j]), 4),
                "jd_id": rows[j]["jd_id"], "jd_job": rows[j]["jd_job"],
                "jd_occ": rows[j]["jd_occ"],
            })
    df = pd.DataFrame(out)
    df.to_csv(CANDIDATES, index=False)
    # Sidecar so every consumer (including the run engine, which can't import
    # this script) can join posting text back on by jd_id.
    jd[["jd_id", "jd_job", "jd_occ", "jd_text"]].to_csv(JD_LOOKUP, index=False)
    print(f"wrote {len(jd)} postings -> {JD_LOOKUP}")
    print(f"wrote {len(df)} candidate pairs -> {CANDIDATES}")
    print("\ntop-1 JD occupation vs resume occupation (cross-occupation is "
          "expected and desirable here):")
    t1 = df[df["rank"] == 0]
    print(pd.crosstab(t1.category, t1.jd_occ).to_string())


# ----------------------------------------------------------------- probing
_lock = threading.Lock()
_local = threading.local()


def _provider():
    if not hasattr(_local, "p"):
        p = get_provider(PROVIDER)
        if PROVIDER == "openai" and hasattr(p, "service_tier"):
            p.service_tier = "default"  # NOT "standard" -- the API rejects it
        _local.p = p
    return _local.p


def _probe(item, repeat_idx, log_path):
    q = QUESTION.format(jd=item["jd_text"])
    # `err` must exist before the falsy-resp branch below: a call that SUCCEEDS
    # but returns an empty string is falsy without any exception having been
    # raised, which crashed the whole run on Kimi K3 (a reasoning model can
    # spend its entire token budget on hidden thinking and return no content).
    err = None
    resp = None
    for attempt in range(5):
        try:
            resp = _provider().infer_text(MODEL, None, item["resume_text"], q, SYSTEM)
            break
        except Exception as e:
            err = f"exception: {e}"
            if attempt < 4:
                time.sleep(1.5 * 2**attempt)
    if resp:
        dec, perr = parse_decision(resp)
    else:
        dec, perr = None, (err or "empty_response")
    rec = {
        "resume_id": int(item["resume_id"]), "category": item["category"],
        "jd_id": item["jd_id"], "jd_job": item["jd_job"], "rank": int(item["rank"]),
        "cosine": float(item["cosine"]), "repeat_idx": repeat_idx,
        "decision": dec, "parse_error": perr,
        "raw_response": (resp or "")[:2000],
    }
    with _lock:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def _run_probes(items, repeats, log_path, workers=8):
    tasks = [(it, r) for it in items for r in range(repeats)]
    recs = []
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_probe, it, r, log_path) for it, r in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="probe"):
            recs.append(f.result())
    return pd.DataFrame(recs)


def _load_texts():
    return {
        json.loads(p.read_text())["resume_id"]: json.loads(p.read_text())["plain_text"]
        for p in (CORPUS / "resumes").glob("*.json")
    }


def cmd_calibrate(args):
    RUNDIR.mkdir(parents=True, exist_ok=True)
    cand = attach_jd_text(pd.read_csv(CANDIDATES))
    texts = _load_texts()
    occupations = sorted(cand.category.unique())

    # Difficulty per occupation = which similarity rank to draw. Rank 0 is the
    # best-fitting JD in the corpus (easiest); higher ranks are progressively
    # worse fits (harder). Start everyone at a mid rank and let the measured
    # accept rate move each occupation independently.
    max_rank = int(cand["rank"].max())
    rank_by_occ = {o: args.start_rank for o in occupations}
    # A fixed step overshoots whenever an occupation's band sits between two
    # rungs: it reads 62% at rank 8, 35% at rank 12, and bounces there until
    # max_iters runs out. Halving the step on every direction reversal turns
    # that oscillation into a bisection, so each occupation converges instead.
    step_by_occ = {o: args.step for o in occupations}
    last_dir = {o: 0 for o in occupations}
    history = []
    calls = 0

    for it in range(1, args.max_iters + 1):
        items = []
        for occ in occupations:
            sub = cand[(cand.category == occ) & (cand["rank"] == rank_by_occ[occ])]
            for _, r in sub.iterrows():
                items.append({**r.to_dict(), "resume_text": texts[r.resume_id]})
        log = RUNDIR / f"calibration_iter{it}.jsonl"
        print(f"\n=== ITER {it}: ranks {rank_by_occ} -> {len(items)} probes ===")
        df = _run_probes(items, repeats=1, log_path=log)
        calls += len(df)
        attempted = len(df)
        df = df[df.decision.notna()].copy()
        # Object dtype (mixed None/int) makes .mean() return objects, which blows
        # up formatting downstream; coerce once here.
        df["decision"] = pd.to_numeric(df["decision"])

        # A partial iteration must not be treated as a measurement. When Kimi K3
        # exhausted its Together credit mid-run, one iteration returned 10 valid
        # decisions out of 234 and the loop calibrated on them without complaint,
        # moving three occupations on essentially no data.
        got = len(df)
        if attempted and got / attempted < 0.8:
            print(f"\nABORT: only {got}/{attempted} probes returned a decision this "
                  f"iteration. Calibrating on a partial sample would move ranks on "
                  f"noise. Check provider errors in {log}.")
            return 1

        rates = df.groupby("category").decision.mean()
        print(rates.mul(100).round(1).to_string())
        history.append({"iter": it, "ranks": dict(rank_by_occ),
                        "rates": rates.round(3).to_dict(),
                        "steps": dict(step_by_occ), "calls": calls})
        (RUNDIR / "calibration_history.json").write_text(json.dumps(history, indent=1))

        done = True
        stuck, off_band = [], []
        for occ in occupations:
            rate = rates.get(occ, 0.0)
            if TARGET_LO <= rate <= TARGET_HI:
                continue
            done = False
            off_band.append(occ)
            # +1 = too easy, move to a worse-fitting JD; -1 = too hard.
            direction = 1 if rate > TARGET_HI else -1
            if last_dir[occ] and direction != last_dir[occ]:
                # Overshot: bisect from here.
                step_by_occ[occ] = max(1, step_by_occ[occ] // 2)
                print(f"  {occ}: overshot, halving step -> {step_by_occ[occ]}")
            elif last_dir[occ] == direction:
                # Still moving the same way -- the target is further off than the
                # current step can reach. Ladders have long plateaus: Sol's SALES
                # sat at 64-74% from rank 40 all the way to 300 before falling to
                # 33% by 700. A step that only ever shrinks crawls; from rank 40
                # with step 16 and 8 iterations it could not pass rank 168, so it
                # burned the whole budget without finding the band. Expand while
                # the direction holds, then bisect once we overshoot.
                step_by_occ[occ] = min(args.max_step, step_by_occ[occ] * 2)
            last_dir[occ] = direction
            moved = min(max_rank, max(0, rank_by_occ[occ] + direction * step_by_occ[occ]))
            if moved == rank_by_occ[occ]:
                # Already at a rail, or the step rounded to nothing.
                stuck.append(occ)
            rank_by_occ[occ] = moved
        if done:
            print(f"\nALL OCCUPATIONS IN [{TARGET_LO:.0%}, {TARGET_HI:.0%}] "
                  f"after {it} iterations, {calls} calls (~${calls*0.006:.2f})")
            pd.DataFrame([{"category": o, "rank": rank_by_occ[o]} for o in occupations]
                         ).to_csv(RUNDIR / "calibrated_ranks.csv", index=False)
            return 0
        # Every occupation still off-band has run out of room to move: more
        # iterations would re-probe the same ranks, so stop and report which
        # ones the corpus cannot bring into the band.
        if off_band and set(stuck) == set(off_band):
            print(f"railed and cannot reach the band: {sorted(stuck)} "
                  f"(ranks { {o: rank_by_occ[o] for o in stuck} })")
            return 1

    print(f"\nnot converged in {args.max_iters} iterations")
    return 1


def cmd_sensitivity(args):
    """Gate 3: keep items whose decision moves when the item is PERTURBED.

    Replaces the repeat-based gate below. Measured on the calibration logs,
    Sonnet 5 is ~95% deterministic at default sampling -- only 5.1% of items
    ever disagreed across up to 5 identical probes, and Engineering managed
    2.6% across five. A gate asking "does this item flip when re-asked" would
    therefore reject a perfectly good item set for having no sampling noise,
    which is not a statement about item quality at all.

    Boundary proximity is the property the experiment actually needs, because
    swapping Arial for Comic Sans is a perturbation, not a resample. Perturbing
    JD difficulty instead, the same items move on 33-72% of cases. So: probe
    each item at its calibrated rank and at two bracketing ranks, and call it
    sensitive if the decision is not constant across the three.

    Reuses every decision already on disk (calibration iterations, earlier
    sensitivity runs) and probes only the gaps.
    """
    ranks = pd.read_csv(RUNDIR / "calibrated_ranks.csv").set_index("category")["rank"]
    cand = attach_jd_text(pd.read_csv(CANDIDATES))
    texts = _load_texts()
    max_rank = int(cand["rank"].max())

    wanted = {}  # (category, resume_id, rank) -> item
    for occ, r0 in ranks.items():
        rungs = sorted({max(0, r0 - args.delta), int(r0), min(max_rank, r0 + args.delta)})
        sub = cand[(cand.category == occ) & (cand["rank"].isin(rungs))]
        for _, r in sub.iterrows():
            wanted[(occ, int(r.resume_id), int(r["rank"]))] = {
                **r.to_dict(), "resume_text": texts[r.resume_id]
            }

    have = set()
    rows = []
    for path in sorted(RUNDIR.glob("calibration_iter*.jsonl")) + [RUNDIR / "sensitivity.jsonl"]:
        if not path.exists():
            continue
        for line in open(path):
            rec = json.loads(line)
            if rec.get("decision") is None:
                continue
            rows.append(rec)
            have.add((rec["category"], int(rec["resume_id"]), int(rec["rank"])))

    todo = [it for k, it in wanted.items() if k not in have]
    print(f"{len(wanted)} item-rank cells needed, {len(wanted) - len(todo)} already "
          f"on disk, probing {len(todo)} (~${len(todo) * 0.006:.2f})")
    if todo:
        new = _run_probes(todo, repeats=1, log_path=RUNDIR / "sensitivity.jsonl")
        rows.extend(new[new.decision.notna()].to_dict("records"))

    d = pd.DataFrame(rows)
    d = d[[(r.category, int(r.resume_id), int(r["rank"])) in wanted
           for _, r in d.iterrows()]]
    # Majority-vote duplicates so the few stochastic items do not masquerade
    # as sensitivity.
    m = d.groupby(["category", "resume_id", "rank"]).decision.mean().round().reset_index()

    per_occ, keep = [], []
    for occ, sub in m.groupby("category"):
        p = sub.pivot(index="resume_id", columns="rank", values="decision").dropna()
        sens = p.nunique(axis=1) > 1
        keep.extend((occ, int(i)) for i in p.index[sens])
        per_occ.append({"category": occ, "n": len(p), "sensitive": sens.mean(),
                        "accept_at_calibrated": float(
                            p[int(ranks[occ])].mean()) if int(ranks[occ]) in p else float("nan")})
    rep = pd.DataFrame(per_occ)
    print("\n=== boundary sensitivity by occupation ===")
    print(rep.round(3).to_string(index=False))
    share = rep.sensitive.mean()
    print(f"\nmean sensitive share {share:.3f} (gate: >= {MIN_FLIPPABLE_SHARE})")

    pd.DataFrame(keep, columns=["category", "resume_id"]).to_csv(
        RUNDIR / "sensitive_items.csv", index=False)
    m.to_csv(RUNDIR / "sensitivity_matrix.csv", index=False)
    if share >= MIN_FLIPPABLE_SHARE:
        print("GATE 3 PASS")
        return 0
    print("GATE 3 FAIL -- widen --delta or recalibrate")
    return 1


def cmd_flippability(args):
    """Superseded by cmd_sensitivity; kept as a diagnostic of sampling noise."""
    ranks = pd.read_csv(RUNDIR / "calibrated_ranks.csv").set_index("category")["rank"]
    cand = attach_jd_text(pd.read_csv(CANDIDATES))
    texts = _load_texts()
    items = []
    for occ, rank in ranks.items():
        sub = cand[(cand.category == occ) & (cand["rank"] == rank)]
        for _, r in sub.iterrows():
            items.append({**r.to_dict(), "resume_text": texts[r.resume_id]})

    log = RUNDIR / "flippability.jsonl"
    print(f"{len(items)} items x {args.repeats} repeats = "
          f"{len(items)*args.repeats} calls (~${len(items)*args.repeats*0.006:.2f})")
    df = _run_probes(items, repeats=args.repeats, log_path=log)
    df = df[df.decision.notna()]

    g = df.groupby(["resume_id", "category"]).decision.agg(["mean", "size"]).reset_index()
    g["flippable"] = (g["mean"] > 0) & (g["mean"] < 1)
    share = g.flippable.mean()
    print("\n=== per-occupation ===")
    print(g.groupby("category").agg(
        accept=("mean", "mean"), flippable=("flippable", "mean"), n=("mean", "size")
    ).round(3).to_string())
    print(f"\noverall accept {g['mean'].mean():.3f}, "
          f"flippable share {share:.3f} (gate: >= {MIN_FLIPPABLE_SHARE})")
    g.to_csv(RUNDIR / "item_flippability.csv", index=False)

    if share >= MIN_FLIPPABLE_SHARE:
        print("GATE 3 PASS")
        return 0
    print("GATE 3 FAIL -- re-run calibration with a finer step")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--flippability", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--delta", type=int, default=8,
                    help="rank offset each side of the calibrated rank (gate 3)")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--refresh-embeddings", action="store_true",
                    help="re-embed instead of reusing the cached vectors")
    ap.add_argument("--start-rank", type=int, default=8)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--max-step", type=int, default=512,
                    help="ceiling on the expanding search step (see cmd_calibrate)")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--model", default="claude-sonnet-5",
                    help="model to calibrate against (calibration is model-specific)")
    # Choices come from the registry so a newly registered provider is usable
    # here without a second edit (this list had drifted and rejected `kimi`).
    from typo_eval.providers import PROVIDERS as _P
    ap.add_argument("--provider", default="anthropic", choices=sorted(_P))
    ap.add_argument("--tag", default="",
                    help="write artifacts to results/.../cal_<tag>/ instead of the run root")
    a = ap.parse_args()
    _configure(a)
    if a.candidates:
        cmd_candidates(a)
    elif a.calibrate:
        raise SystemExit(cmd_calibrate(a))
    elif a.sensitivity:
        raise SystemExit(cmd_sensitivity(a))
    elif a.flippability:
        raise SystemExit(cmd_flippability(a))
    else:
        ap.error("pick --candidates | --calibrate | --sensitivity | --flippability")
