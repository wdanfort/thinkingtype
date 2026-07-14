# Gate-drift monitor: regression testing for AI gate decisions

If a model makes yes/no gate decisions in your product (screen this résumé?
remove this comment? approve this claim?), then three things silently change
your policy without any code change: **model upgrades**, **prompt edits**,
and **switching how content reaches the model (text vs. rendered image)**.
This kit measures all three at the place where they actually matter — the
decision boundary.

## Why boundary calibration is the point

Evaluating a gate on typical cases will tell you it's robust, because most
cases are easy and every model version agrees on them. Changes show up on
the *borderline* cases — which is where appeals, complaints, and audits
live. This harness finds each model's own borderline cases first (graded
"strength ladder" stimuli + repeated text-mode sampling), then measures
changes there. Effects invisible in average-case evals (10–18pp decision
shifts) are directly measurable at the boundary.

## The workflow

```bash
# 1. Write stimuli (built-in banks: resume / moderation / appeal)
typo-eval --config configs/my_gate.yaml gates-build     # or bring your own CSV

# 2. Render document images for the visual variants in the config
typo-eval --config configs/my_gate.yaml gates-render

# 3. Calibrate: repeated text-mode sampling -> boundary item selection
typo-eval --config configs/my_gate.yaml gates-calibrate --provider anthropic

# 4. Vision arm on the selected boundary items
typo-eval --config configs/my_gate.yaml gates-run --provider anthropic

# 5. Per-run analysis (modality deltas, variant effects, fairness contrast)
typo-eval --config configs/my_gate.yaml gates-analyze

# 6. Drift: compare any two runs
typo-eval --config configs/my_gate.yaml gates-drift --run-a gates_v1 --run-b gates_v2
```

`gates-drift` reports, per provider × gate: decision flips on shared items
(favorability-signed, so "harsher"/"softer" is explicit), mean p(yes) shift,
boundary migration, flipped-item lists, and — when both runs have vision
arms — the change in image-vs-text drift per visual variant. Outputs land in
`results/gates/drift_<A>_vs_<B>_<timestamp>/`.

## The three regression scenarios

1. **Model upgrade.** Copy your config, bump `model_text`/`model_vision`
   and `run_tag`, rerun steps 3–5, then `gates-drift`. Flipped items are
   your release notes for the policy change you didn't know you shipped.
2. **Prompt change.** Point `inference.prompt_variant` at a new prompt
   table entry (or a custom gate definition), rerun, compare. (Measured
   example: a written-rubric prompt changed 2–6% of text decisions while
   removing most image-vs-text drift.)
3. **Vendor swap.** Same run, different `--provider`; or drift-compare two
   single-provider runs. Directional drift differs by vendor — treat a swap
   as a policy review, not a drop-in.

## Bringing your own gate

Define it in the config; no code changes:

```yaml
gates:
  - warranty_claim

custom_gates:
  warranty_claim:
    system_prompt: |
      You review warranty claims. Approve when: (a) purchase within 24
      months; (b) defect is a covered failure, not wear or damage; (c) no
      prior claim for the same defect. Answer only with Yes or No.
      Do not explain your answer.
    question: "Should this claim be approved under the stated criteria?"
    yes_is_favorable: true
    stimulus_path: data/inputs/gates/warranty_claim.csv
```

The stimulus CSV needs columns `item_id, gate, scenario, level, text`.
Write each scenario as a graded ladder (level 1 = clearly deny … level N =
clearly approve): calibration finds where your model's boundary falls on
each ladder, so you don't need to guess which cases are borderline. Keep
stimuli free of names and demographic signals so decision drift can't be
confounded with fairness effects — and add a fairness variant (e.g.
OpenDyslexic vs. sans) if you want the format-neutrality check.

## Reading the numbers

- Everything favorability-signed: positive = more favorable to the person
  judged. "Net_fav −4" means four net decisions got harsher.
- `p_yes` granularity is 1/repeats; frontier models are nearly
  deterministic at default sampling, so boundary items are mostly
  flip-adjacent ladder steps rather than 50/50 items.
- n=20 boundary items per gate resolves effects of roughly ≥5pp; scale
  `n_select` and the stimulus bank for finer effects (the fairness follow-up
  used 40 ladders / 60 selected items to pin a ~4pp effect).

## Findings that motivated this tool

See `results/gates/gates_v1/REPORT.md` and `results/gates/BATCH1_REPORT.md`:
moderation gates are 10–13pp more permissive on images at the boundary;
résumé screening drifts in opposite directions by vendor; written rubrics
eliminated the résumé drift entirely (0 disagreements in 480 boundary
calls) and halved the moderation drift; screenshot-style UI chrome shifts
the drift in provider-specific directions; accessibility-font content
drifted only ever harsher (9 of 10 moved items) across two providers.
