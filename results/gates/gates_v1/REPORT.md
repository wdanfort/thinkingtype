# Gates v1: does the text→image switch move real decisions at the boundary?

**Run:** `gates_v1` (July 2026). Models: GPT-5.5, Claude Sonnet 5, Gemini 3.5 Flash
(same model for text and vision within each provider). 4,608 text calibration
calls + 9,720 vision calls, zero parse failures.

## What this run was for

v0b's Finding 5 couldn't test whether the text/image gap moves *decisions*
because both stimulus sets missed the decision boundary (everyday sentences:
~0% escalation; alarming sentences: ~93%). This run fixes that with a
calibration loop: 192 hand-built items on graded strength ladders across three
real gates (résumé screen, moderation removal, hardship-grant appeal), each
model's text-mode p(yes) measured with 8 repeated samples per item, and the 20
items per gate nearest each model's own boundary selected for the vision arm
(9 visual variants × 6 repeats).

**Boundary calibration succeeded:** mean text p(yes) of selected items is
0.48–0.53 in every provider × gate cell, and every scenario ladder crosses the
model's flip point. The vision arm ran exactly where decisions are movable.

## Finding 1: the modality switch moves boundary decisions, and the direction is gate- and provider-specific

Reference fonts only (plain serif/sans), item-clustered bootstrap, n=20
boundary items per cell. Δfavorability = p(image) − p(text), signed so
positive favors the person being judged:

| Gate | GPT-5.5 | Sonnet 5 | Gemini 3.5 Flash |
|---|---|---|---|
| Moderation | **+0.100 [0.008, 0.217]** | **+0.131 [0.013, 0.281]** | +0.027 [0.000, 0.060] |
| Résumé | **−0.042 [−0.092, −0.002]** | −0.010 (ns) | **+0.156 [0.035, 0.304]** |
| Appeal | +0.021 (ns) | −0.017 (ns) | −0.025 (ns) |

- **Moderation leniency as images (GPT-5.5, Sonnet 5):** borderline
  harassment that the same model removes as text survives as an image
  ~10–13pp more often. Every majority-vote flip in these cells (6/6 Sonnet,
  4/4 GPT-5.5) went the same direction: toward *keep*. This is the v0b
  "quiet drop" concern, now confirmed at a real gate with adequate power —
  vision-mode review is systematically more permissive at the moderation
  boundary.
- **Résumé screening splits by provider:** Gemini advances boundary
  candidates ~16pp more often as images (8/8 flips toward advance, 20% of
  boundary items flip on modality alone); GPT-5.5 advances slightly fewer.
  Same content, opposite directions — consistent with v0b's finding that
  disagreement direction is learned, not architectural.
- **The rubric-backed appeal gate is robust** in all three models (|Δ| ≤
  2.5pp). When the decision has explicit criteria in the prompt, the
  modality switch stops mattering. That's an actionable mitigation, not
  just a null.

## Finding 2: the OpenDyslexic penalty does not survive at the decision gate

Within-image contrast (OpenDyslexic vs same-content sans reference), pooled
over gates, n=60 items/provider: GPT-5.5 +0.011 [−0.022, +0.056], Sonnet 5
−0.017 [−0.094, +0.061], Gemini −0.025 [−0.072, +0.014]. No significant
shift for any provider; the only lean (moderation, Sonnet −0.10, Gemini
−0.075, both CIs touching 0) points toward accessibility-font comments being
removed slightly more often, and would need a dedicated larger run to
confirm. **On decisions, unlike the v0b judgment dimensions, accessibility
formatting does not measurably change outcomes at this sample size.** Large
print (A2) was, if anything, *more* favorable (Gemini +0.097, significant).

## Finding 3: beyond fonts, no visual variant beats the modality switch itself

Across color (red, low-contrast gray), yellow highlighting, cramped layout,
large print, comic, and OpenDyslexic: within-image variant effects are small
(mostly |Δ| < 0.05, ns) relative to the text→image switch. Gemini is the
exception — uniformly more favorable in image mode across variants (large
print +0.097, highlight +0.089, sans +0.075, all significant pooled) — i.e.,
its shift is modality-level, not typography-level.

## What changes for builders

1. **VLM moderation pipelines have a measurable permissive bias at the
   boundary.** If content enters review as screenshots/images, borderline
   harassment survives more often than the same pipeline would allow as
   text. Gate on extracted text, or shadow-test image vs text on your own
   boundary cases.
2. **VLM résumé screening is provider-lottery at the boundary** (direction
   flips between vendors). Don't let a vision model make the advance/reject
   call on scanned documents without a text-extraction cross-check.
3. **Explicit rubrics in the prompt appear protective.** The appeal gate —
   the only one with criteria spelled out — was immune in all three models.
4. **Accessibility-fairness:** no compliance-level effect on decisions found
   here; the moderation lean is the one cell worth a scaled follow-up.

## Limitations

- Frontier text models are nearly deterministic at default sampling, so
  "boundary" items are mostly flip-adjacent on the ladder (deterministic No
  next to deterministic Yes) rather than 50/50-sampled; p-resolution per
  item is 1/8 (text) and 1/6 (image).
- n=20 items per provider×gate; effects under ~5pp are not detectable.
- Synthetic single-page documents; no names or demographics by design.
- One prompt per gate; prompt-sensitivity untested (though the rubric
  result suggests it matters).

## Reproduce

```
typo-eval --config configs/gates_v1.yaml gates-build
typo-eval --config configs/gates_v1.yaml gates-render
typo-eval --config configs/gates_v1.yaml gates-calibrate --provider <p>
typo-eval --config configs/gates_v1.yaml gates-run --provider <p>
typo-eval --config configs/gates_v1.yaml gates-analyze
```

Artifacts: `analysis/summary.md` (full tables), `analysis/paired_items.csv`
(item-level), `analysis/figures/` (forest plots per gate, text-vs-image
scatter), `selection/selected_<provider>.csv` (boundary sets with text
p(yes)).
