# Batch 1: rubric shield, screenshot realism, fairness scale-up

**Runs:** `gates_v1_rubric`, `gates_v1_realism`, `gates_v1_fair` (July 2026).
Models: Claude Sonnet 5, Gemini 3.5 Flash, and GPT-5.5 complete on all three
experiments. (The OpenAI account exhausted its quota mid-batch; all GPT-5.5
legs were rerun clean after the billing fix, and the tables below reflect the
completed reruns.)

All effects are Δfavorability = p(yes as image) − p(yes as text), signed so
positive favors the person judged, item-clustered bootstrap 95% CIs, on
boundary-calibrated items. Baselines referenced from `gates_v1/REPORT.md`.

## 1. Rubric shield (#1): writing the criteria into the prompt largely closes the modality gap

Method: same two gates that moved in gates_v1 (moderation, résumé), same
stimulus banks, but the system prompt now spells out the decision criteria.
The boundary was **recalibrated under the rubric prompt** (so each prompt is
tested at its own boundary — the comparison is "how much does the modality
switch move this gate," not "same items twice").

| Gate | v1 prompt (gates_v1) | Rubric prompt | Verdict |
|---|---|---|---|
| Résumé, Gemini | **+0.156** [+0.04, +0.30] | **0.000** [0, 0] | eliminated |
| Résumé, Sonnet | −0.010 (ns) | **0.000** [0, 0] | stays null |
| Résumé, GPT-5.5 | **−0.042** [−0.09, −0.00] | −0.048 [−0.13, +0.00] | unchanged (now marginal) |
| Moderation, Sonnet | **+0.131** [+0.01, +0.28] | **+0.065** [+0.01, +0.13] | halved, not eliminated |
| Moderation, Gemini | +0.027 (ns) | −0.027 (ns) | stays null |
| Moderation, GPT-5.5 | **+0.100** [+0.01, +0.22] | **+0.079** [+0.02, +0.15] | essentially unchanged |

The Sonnet/Gemini résumé result is striking and not degenerate: the
rubric-selected items still straddle the boundary (mean text p(yes) = 0.50,
deterministic No/Yes items adjacent to flip points), and the vision arm
agreed with the text arm **on every one of 40 items × both providers (480
image calls, zero disagreements)**. Under the terse v1 prompt the same
conditions produced 12–20% majority flips.

But GPT-5.5 revises the headline: its drift passes through the rubric
essentially unchanged on both gates (résumé point estimate −0.042 → −0.048;
moderation +0.100 → +0.079, CI still excluding zero). **The rubric shield is
itself provider-dependent** — a complete fix for two models, close to no fix
for the third. Sonnet's moderation case sits in between: halved but not
eliminated, consistent with "person vs. argument" being a criterion that
still requires judgment.

**Product takeaway:** writing the decision criteria into the prompt is free
and removed most or all drift in two of three models — do it. But it is not
a guarantee; whether it works for *your* model on *your* gate is exactly
what the drift monitor (see `docs/GATE_DRIFT.md`) measures in an afternoon.

## 2. Screenshot realism (#3): the leniency is largest for sterile renders; UI chrome shrinks it

Method: the gates_v1 boundary moderation items re-run as images inside
forum/phone UI chrome (avatar, neutral synthetic handle, timestamp,
reply/report buttons; light, dark, and phone-status-bar styles), against the
plain document render. Text baseline and item selection reused verbatim from
gates_v1 (same prompt, same items).

| Image style | Sonnet 5 | Gemini 3.5 Flash | GPT-5.5 |
|---|---|---|---|
| Plain document (T3) | **+0.181** [+0.03, +0.34] | +0.065 [0.00, +0.16] | **+0.083** [+0.02, +0.17] |
| Forum card, light | −0.002 [−0.12, +0.13] | +0.090 [0.00, +0.20] | **+0.142** [+0.04, +0.27] |
| Forum card, dark | +0.065 [−0.06, +0.22] | +0.090 [0.00, +0.22] | **+0.108** [+0.02, +0.22] |
| Phone screenshot | +0.023 [−0.15, +0.20] | +0.056 [0.00, +0.13] | **+0.108** [+0.01, +0.24] |

The three providers respond to realism in three different ways: Sonnet's
leniency shrinks toward zero under chrome, Gemini's is small and flat
regardless, and **GPT-5.5's grows** — realistic screenshots are its *worst*
case, and its chrome CIs all exclude zero. An earlier two-provider interim
framing ("the worst case is the sterile render") did not survive the third
provider; the durable finding is:

1. **Visual context is an active policy variable.** How content is
   packaged into the image (plain re-render vs. screenshot with UI
   affordances) measurably changes the removal decision in every provider —
   in a provider-specific direction.
2. **No provider's moderation leniency disappears under realism.** The
   modality risk is not an artifact of sterile lab renders.

Reliability note: the plain-render condition is an independent resample of
the gates_v1 condition and reproduced it (+0.181 vs +0.181 Sonnet; +0.083
vs +0.100 GPT-5.5, T3-only vs T1+T3). Mechanism remains speculation — e.g.
Report/vote affordances may cue "content under moderation" for Sonnet while
GPT-5.5 may read casual-forum framing as license for rougher speech norms —
and would need a targeted follow-up to pin down.

## 3. Fairness scale-up (#4): the OpenDyslexic-harsher lean survives at n=60, small and one-directional

Method: moderation bank expanded from 12 to 40 threads (200 items), 60
boundary items per provider, within-image contrast of OpenDyslexic vs. sans
on identical content (this isolates the font; modality is held constant).

- GPT-5.5: mean Δfavorability **−0.061 [−0.122, −0.008]** — CI excludes
  zero. 11 items moved, 8/11 toward removal of the OpenDyslexic version.
- Gemini: **−0.039 [−0.089, −0.003]** — CI excludes zero. 4 items moved,
  4/4 toward removal.
- Sonnet: −0.039 [−0.108, +0.028] — CI includes zero. 6 items moved, 5/6
  toward removal.
- Pooled across providers: **17 of 21 moved items moved toward removal**
  (two-sided sign test p ≈ 0.007, with the caveat that the three providers
  saw overlapping item sets, so the pooled test overstates independence).

Honest framing: the effect is sparse (7–18% of boundary items) and small in
aggregate (4–6pp), and per-provider sign tests don't individually reach
significance. But the direction is consistent across all three providers,
two sample sizes, and two stimulus banks, and two of three clustered CIs
exclude zero — comments rendered in an accessibility font are never treated
*more* leniently at the boundary, only harsher. This has graduated from
"curiosity" to "specific, testable compliance question": anyone moderating
rendered content should include accessibility-formatted variants in their
evaluation set ("format neutrality" as a tested property).

## Corrections and operational notes

- An earlier in-session status said the GPT-5.5 realism leg "finished
  clean" — that was wrong (line count checked, parse status not); all 480
  of its records were quota errors. All GPT-5.5 Batch-1 legs were rerun
  after the OpenAI account was refunded credit (`gates-calibrate` +
  `gates-run --provider openai` for `gates_v1_rubric` and `gates_v1_fair`;
  `gates-run --provider openai` for `gates_v1_realism`).
- Poisoned records were stripped from all JSONLs (resume treats logged
  errors as done, so they must not remain), and the selection file derived
  from partial data was deleted.
- New reusable machinery this batch: `prompt_variant` (rubric prompt table),
  UI-chrome renderer (`gates/chrome_render.py`), per-run render metadata,
  pinned font normalization, variant-filtered vision runs, and a 40-thread
  moderation bank.
