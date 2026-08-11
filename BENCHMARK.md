# ThinkingType Benchmark — design specification

**Status: draft, v4-era.** Reflects what hrsim_v4 established (Aug 2026). Numbers
marked *(measured)* come from the runs in `results/hrsim/hrsim_v4/`.

---

## What this measures

Two numbers per model. Both are useful; only the second is about typography.

| # | Metric | What it tells you |
|---|---|---|
| 1 | **Screening leniency** — where the model's decision boundary sits on a fixed résumé×job ladder | How readily a model advances candidates. A model property, independent of presentation. |
| 2 | **Presentation sensitivity** — how much the model's decision moves when the *same résumé* is re-rendered in a different typeface, or shown as an image instead of text | Whether presentation leaks into a hiring decision. |

Metric 1 is not scaffolding for metric 2 — it is a finding in its own right, and it
is the one that does not saturate. *(Measured: on identical items, Sonnet 5 accepts
48.3% and GPT-5.6 Sol accepts 73.9%.)*

---

## Core design principle: freeze the ladder, not the item set

Presentation effects can only be measured where a decision has room to move. At the
floor, every item is a confident "no" and a null result is uninterpretable — you
cannot distinguish "no effect" from "no room". *(hrsim_v2/v3 sat at 4–12% accept and
measured nothing.)*

But every model's boundary sits in a different place, and boundaries move as models
change. So the benchmark ships **the ladder**, not a fixed set of résumé–job pairs:

- 234 pseudonymised résumés, frozen and content-hashed
- 8 rendered arms per résumé (or the deterministic render pipeline + bundled fonts)
- the full JD corpus (3,553 deduplicated postings)
- for each résumé, a **ranked list of every posting** by hubness-corrected embedding
  similarity — the difficulty ladder
- the screening prompt, verbatim

Each model is then measured near **its own** boundary, discovered by the calibration
protocol below. The item set is a function of the model; the *manipulated object* —
the rendered résumé — is byte-identical for everyone.

### Why difficulty is tuned via the job, not the résumé

Walking up the ladder changes **which posting the résumé is judged against**. Rank 0
is the best-matching job (easy yes); rank 200 is a poor match (easy no). The résumé
itself never changes.

This is load-bearing. Tuning difficulty by editing the résumé — shortening it,
weakening its credentials — would alter the very artifact whose *rendering* is under
test, contaminating the comparison. Moving the job context keeps the manipulated
object frozen and puts all difficulty variation in the surrounding context.

### Headroom against saturation

The published ladder exposes the top 200 of 3,553 postings per résumé. A model too
lenient at rung 200 gets a deeper ladder, regenerated for free from the cached
embeddings — roughly 15× of runway before the corpus itself needs replacing.

---

## Calibration protocol (per model, required)

Boundary calibration **does not transfer between models**. *(Measured: the item set
calibrated to put Sonnet 5 at 40–60% puts GPT-5.6 Sol at 73.9%.)* Any multi-model
comparison that calibrates once and reuses the item set is silently comparing models
at different operating points.

The protocol is deterministic and seeded, not "tune until it looks right":

1. Probe **all items in each occupation** at a starting rung; measure accept rate
   **per occupation**.
2. Move each occupation's rung independently. **Double the step while the direction
   holds; halve it on the first reversal, then bisect.**
3. Stop when every occupation is within **40–60%**, or when every still-off-band
   occupation has run out of ladder (report which).

Cost: ~5 iterations × 234 items ≈ **1,200 calls**.

> **Do not subsample.** An earlier draft of this spec called for a 60-item subsample
> to save money. That is unworkable: calibration runs **per occupation**, so 60 items
> is 10 per occupation — a standard error of ~16pp against a 20pp-wide band. Even at
> the full 39 per occupation the read noise is ~8pp, and the loop can be seen chasing
> it *(measured: Engineering read 0.49–0.59 across seven probes at the same rung)*.
> Calibration is the cheap part of a run; buying precision here is worth it.

> **Both search rules are load-bearing.** *(Measured: a halve-only step starting at
> rung 40 with step 16 and 8 iterations cannot pass rung 168. Sol's SALES band is at
> rung 400, so the loop burned its entire budget and reported the occupation as
> unreachable — it was not.)*

**Always publish the achieved accept rate next to every result.** A model that cannot
be brought into band is a reportable outcome ("out of band"), not a silent comparison
at the wrong operating point.

### Rank is scaffolding, not a result

Publish the calibrated rank for reproducibility only. **Never compare rank across
models or occupations.** Rank is an ordinal position in a similarity-sorted list, and
similarity is heavily compressed, so equal rank steps are not equal difficulty steps.
*(Measured: Sol's SALES sits at 64–74% accept from rung 40 all the way to 300, then
falls to 33% by 700 and 0% at the corpus floor — a long plateau followed by a cliff.
Healthcare reaches the same 50% accept at rung 40 that Sales reaches at rung 400.)*

Every comparative statement keys off **measured accept rate**, never rung number.

---

## Metrics

### 1. Excess disagreement over the model's own noise floor — the headline

Every run includes **one duplicate of the reference arm**. The headline number is:

> **excess = between-arm disagreement − within-arm disagreement**

*(Measured, Sonnet 5: between-arm 7.6% [5.6, 10.1], within-arm 5.6% [3.0, 8.5],
excess **+2.1pp**, CIs heavily overlapping.)*

This is the single most important design decision in the benchmark. It auto-corrects
for temperature, sampling determinism, and vendor differences in decoding — none of
which are the thing under test. Without it, a model that simply resamples more is
indistinguishable from one that is presentation-sensitive.

Cost: one extra arm, ~11% of run cost.

### 2. Effects in log-odds, not percentage points

A 5pp shift at 50% accept and a 5pp shift at 90% are very different perturbations of
the decision function. Log-odds makes them comparable, which matters precisely when a
model lands off-band. *(Measured: the Sol:Sonnet gap is 6.0× in percentage points and
6.9× in log-odds — the vendor difference is not an artifact of operating point.)*

Report percentage points too; they are what a product audience reads.

### 3. Modality and typeface reported separately

These are different phenomena with different implications and must not be collapsed:

- **Modality** — rendered image vs. plain text, holding typeface at the sans reference
- **Typeface** — each typeface vs. the sans reference, holding modality at image

*(Measured, Sol: a significant modality effect — every rendered arm is harsher than
text, −3.0 to −5.1pp, significant for Comic Neue, Roboto Light, and OpenDyslexic —
with **no** significant typeface effect. The two axes genuinely dissociate.)*

### 4. Ink coverage as a control axis

Three weights of one typeface (Roboto Light/Regular/Bold) bracket the five test
typefaces on measured ink coverage *(0.0255–0.0665 vs. 0.0325–0.0532)*. This
separates "how much dark ink is on the page" from "which typeface it is" — the
standing objection to every result in this literature.

*(Measured, Sonnet 5: no decision-level ink effect resolvable at this n — slope CI
[−31, +142] pp per unit coverage — but a clean monotone effect on deliberation
length, Bold − Light = +28.2 tokens [+9.0, +47.5], surviving page-count matching.)*

---

## Two designs, both worth running

Neither controls everything; they fail in different directions, which is why running
both is worth more than choosing one.

| Design | Holds constant | Confounded by |
|---|---|---|
| **Fixed item set** | job pairing, difficulty | each model sits at a different point on its own response curve |
| **Per-model calibration** | operating point (~50% each) | job context differs between models |

Report both. If an effect survives both, the operating-point objection is closed.

### The response curve is the design, not an upgrade

Run **three rungs per model** and report the effect **interpolated to exactly 50%
accept**. Every model is then compared at an identical operating point even though
none was run exactly there.

This was drafted as an optional gold-standard variant. The v4 Sol legs show it is the
minimum viable design, for three converging reasons:

1. **One operating point cannot resolve the effect.** *(Measured: MDE ≈ 5.4pp at
   n=234; the effects in play are 1–5pp. Sol's modality effect measured −4.7pp on one
   item set and −1.3pp on another, and a clustered bootstrap of the difference between
   those two runs spans zero — the runs are indistinguishable from each other AND the
   second is indistinguishable from zero.)*
2. **A single point is confounded by where it sits.** *(Measured: the same model on a
   set calibrated for a different model reads −5.1pp; on its own, −2.6pp. The
   mis-calibration inflated the effect rather than suppressing it.)*
3. **You cannot assume a rung is where you think.** Plateaus mean a chosen rung may
   sit far from the transition — see the SALES trace above.

A curve dissolves all three: the operating point stops being a nuisance parameter and
becomes the independent variable, and the slope estimate pools ~3× the observations.

**Effect magnitude may itself depend on the operating point.** *(Measured, Sol, four
points ordered by accept rate: Healthcare 48.7% → +0.9pp; pooled 53.0% → −1.3 to
−2.6pp; Sales 59.0% → −6.8pp; Sonnet-calibrated 73.9% → −4.7 to −5.1pp.)* Roughly
monotone, and mechanically plausible — a model inclined to accept has more room to be
pushed down. No individual contrast is resolved, which is precisely why the curve is
needed to settle it.

---

## Tiers

| Tier | Items | Arms | Calls | Purpose |
|---|---|---|---|---|
| **Core** | 120 | text, sans, Comic, OpenDyslexic, sans-repeat | ~900 incl. calibration | Routine tracking. Cheap enough that nobody skips calibration. |
| **Full** | 234 | + serif, mono, 3-rung ink ladder | ~2,400 | Publication; includes the ink control axis. |

*(Measured: the full tier on Sonnet 5 cost $28.88 for 2,106 calls — 11.8M input
tokens, dominated by image tokens. OpenDyslexic is the most expensive arm at $3.72,
~50% above serif, because it needs 3.1 pages to serif's 1.97.)*

---

## Reporting requirements

Every published result must carry:

1. Model ID and provider, exactly as sent
2. Achieved calibration rank **and** accept rate per occupation
3. The within-arm noise floor measured in the same run
4. Effects in both percentage points and log-odds, with resume-clustered bootstrap CIs
5. Refusal and parse-failure counts
6. Benchmark version and corpus content hash

---

## Known limitations

- **Contamination.** The résumé corpus (LiveCareer, via Kaggle) is public and will
  enter training data. A leaderboard needs a held-out private split; the public split
  should be versioned so results stay interpretable as leakage accumulates.
- **Identity is deliberately held constant.** All 36 given names are androgynous and
  Anglo, so the benchmark measures presentation *only*. Name and race effects require
  a separate re-render (see hrsim_v2, which used Gaddis-2017 racialized names).
- **Boundary ≠ maximal sensitivity.** Escaping the floor is a necessary condition for
  the measurement to be interpretable. It is *not* established that effects are
  largest at 50% — *(measured: Sol's significant effect appeared at 73.9% accept,
  while Sonnet at 48.3% showed none on decisions)*. 50% is a comparability
  convention, not a proven optimum.
- **Single-repeat runs cannot measure per-item flip rates.** Frontier models are
  ~94–95% deterministic at default sampling *(measured: 5.1% disagreement across up
  to 5 identical probes)*, so repeats buy far less than item count. Spend on items.
- **Perturbation sensitivity is a weak predictor of font sensitivity.** *(Measured:
  items flagged boundary-sensitive by JD perturbation showed arm disagreement 27.3%
  vs. 17.9% for content-locked items.)* Do not restrict a run to the "sensitive"
  subset — the filter is not reliable enough.

---

## Provenance of the design decisions

Each of these was a mistake made and corrected during hrsim_v1–v4; they are recorded
so the benchmark does not repeat them.

| Decision | Why |
|---|---|
| Search the full JD corpus, not curated occupation pools | The hrsim2 allowlists keep 81 of 159 job titles and drop `administrative assistant`, `receptionist`, and 76 others — exactly the plausible targets for support-role résumés. With them filtered out, 16 of 39 healthcare résumés matched "shop assistant". |
| Hubness-correct the similarity scores | Raw cosine lets short generic postings sit near everything: "hotel receptionist" was the best match for 19 of 39 finance résumés. Centering (subtract each posting's mean similarity across résumés) halved the largest attractor and *raised* within-occupation matching. |
| Gate on perturbation sensitivity, not repeat-flipping | A gate asking "does this item flip when re-asked" rejects a good item set for having no sampling noise. Measured repeat-flip rate was 5.1% against a 50% gate; perturbation sensitivity on the same items was 59.4%. |
| Calibrate per occupation, not globally | One similarity percentile cannot put six occupations at the same accept rate. *(Measured spread at a single rung: 41–90%.)* |
| Pin the prompt to the calibration | The accept band only holds under the wording it was measured with. `hrsim4/prompts.py` fails loudly if the runner's prompt drifts from the calibration script's. |
