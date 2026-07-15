# Gates v2: the same protocol, one model release later

**Run:** `gates_v2` (July 14, 2026). Models: Claude Fable 5 (`claude-fable-5`),
GPT-5.6 Sol (`gpt-5.6-sol`) — each vendor's newest GA vision model, replacing
Claude Sonnet 5 and GPT-5.5 from `gates_v1`. Gemini not rerun (3.5 Flash is
still the newest GA tier). Per provider: 2,656 text calibration calls + 3,240
vision calls; zero parse failures, zero structured refusals, zero safety
fallbacks/reroutes (all Anthropic responses served by `claude-fable-5`).

Protocol identical to gates_v1 (boundary calibration at 8 samples/item, 20
items/gate selected per provider, 9 visual variants × 6 repeats), with fresh
calibration so each model is tested at its own boundary. Cross-release
comparison in `results/gates/drift_gates_v1_vs_gates_v2_20260715_025938/`.

## Modality effects at the boundary

Δfavorability = p(yes as image) − p(yes as text), positive favors the person
judged, item-clustered bootstrap 95% CIs, reference + variant fonts pooled as
in gates_v1 tables (n=40 item×arm cells):

| Gate | Claude Fable 5 | GPT-5.6 Sol |
|---|---|---|
| Moderation | −0.060 [−0.177, +0.061] (ns) | +0.046 [−0.054, +0.143] (ns) |
| Résumé | +0.071 [0.000, +0.163] (marginal) | +0.056 [−0.007, +0.123] (marginal; sign test p=0.021) |
| Appeal | −0.002 [−0.034, +0.028] | +0.017 [0.000, +0.042] |

## Cross-release findings (vs. gates_v1)

1. **The moderation image-leniency shrank at both vendors.** Anthropic:
   +0.131 [+0.013, +0.281] (Sonnet 5) → −0.060 ns (Fable 5) — the point
   estimate now leans *harsher* on images, and the shared-item drift table
   shows the plain-sans condition falling from +0.44 to +0.06. OpenAI:
   +0.100 [+0.008, +0.217] (GPT-5.5) → +0.046 ns (Sol), roughly halved.
   gates_v1's headline risk — borderline harassment surviving as
   screenshots — got measurably smaller in one release cycle.
2. **The résumé drift flipped direction at OpenAI and appeared at
   Anthropic.** GPT-5.5 advanced boundary candidates *less* as images
   (−0.042, CI excluding zero); Sol advances them *more* (+0.056, sign test
   p=0.021, bootstrap CI grazing zero). Fable 5 shows the same positive
   lean (+0.071, CI touching zero) where Sonnet 5 was null. Builders who
   characterized their vendor's résumé bias on the previous release now
   have it wrong in direction or existence.
3. **The rubric-backed appeal gate is robust in its fourth and fifth
   models.** |Δ| ≤ 2.5pp in every model tested across two generations.
   Explicit criteria in the prompt remain the most durable mitigation
   this program has found.
4. **Text-mode boundaries are stable across the release** (0–3 item flips
   per provider×gate on shared items): what changed is how the new models
   treat images, not where they draw the underlying line.
5. **Operational:** Fable 5's new structured-refusal machinery
   (`stop_reason: "refusal"`, Opus-4.8 fallback) never triggered on these
   synthetic boundary stimuli; harness refusal-tagging ran but recorded
   nothing.

## Limitations

Same as gates_v1 (deterministic sampling → flip-adjacent boundary items,
n=20 items/gate/provider, synthetic stimuli, one prompt per gate), plus:
both marginal résumé CIs touch zero — treat the direction, not the
magnitude, as the finding until a scaled rerun.

## Reproduce

```
typo-eval --config configs/gates_v2.yaml gates-build
typo-eval --config configs/gates_v2.yaml gates-render
typo-eval --config configs/gates_v2.yaml gates-calibrate --provider <p>
typo-eval --config configs/gates_v2.yaml gates-run --provider <p>
typo-eval --config configs/gates_v2.yaml gates-analyze
typo-eval gates-drift --run-a gates_v1 --run-b gates_v2
```
