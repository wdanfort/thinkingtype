# ThinkingType — Framing Revision Summary

This summary documents the framing revision applied to `docs/index.html` and `docs/index.md`. **No data, methodology, or limitations were changed.** Only framing, sequencing, and the addition of a Related Work section.

## Branch

The task description requested a feature branch named `framing-revision-2026-05`. The harness (system instructions) designated `claude/framing-revision-thinkingtype-Qn7ul` for development on this repo, so changes were made on that branch instead. The branch was **not pushed** per explicit instruction. If you want the branch renamed to match the original spec, run:

```bash
git branch -m claude/framing-revision-thinkingtype-Qn7ul framing-revision-2026-05
```

## Files changed

- `docs/index.html` — primary GitHub Pages view (hero, interactive demo, research content)
- `docs/index.md` — parallel Markdown version of the research content

The interactive demo (font picker, flip-rate chart, JS data block) and all figure references were left intact.

## What changed and why

### 1. Headline + meta tags (overstated → bounded)

- **Before:** "Font choice flips AI model judgments 20–50% of the time"
- **After:** "Vision pipelines deflect borderline cases away from human review — and typography modulates the bias"
- **Why:** The old headline conflated three distinct effects (vision-vs-text divergence, typography modulation, and the cherry-picked trustworthiness dimension) into a single inflated range. The new headline leads with the most practically important finding — the decision asymmetry — and explicitly positions typography as a *modulator* rather than the primary driver.
- The HTML `<title>`, `<meta name="description">`, Open Graph and Twitter Card tags were updated to match. The OG image was also swapped from `comparison_overall.png` to `comparison_decision_flip.png` so social-card previews surface the decision-asymmetry chart.

### 2. TL;DR (single muddled sentence → three nested findings)

The TL;DR now explicitly explains the three-level structure of the result:

1. Vision–text divergence (16–22% baseline, before any typography)
2. Decision asymmetry (73–100% directional toward "don't escalate" for OpenAI/Google; Anthropic neutral)
3. Typography modulation (~5pp on the flip rate; trustworthiness is an outlier dimension at ~50%)

This ordering goes from "what we measured" → "what's practically concerning" → "the colorful angle," instead of starting with the colorful angle and burying the practical finding.

### 3. Hero stats (cherry-picked → bounded)

- Replaced "16–22% — Average dimension flip rate / ~50% — Trustworthiness flip rate / 3 providers / ~10k comparisons"
- With: "16–22% (vision–text divergence on identical content) / 73–100% (decision flips toward 'don't escalate', OpenAI & Google; Anthropic neutral) / ~5pp (max typography modulation) / 3 providers"
- **Why:** Each stat is now self-bounding — the label tells you what subset of the data the number describes, so it can't be read as a global "fonts flip 50% of judgments" claim.

### 4. Hero Comic Sans / Times Roman example (oversold → labeled illustrative)

The visual element is preserved, but the caption now says: "An illustrative example of the kind of flip we observe. Trustworthiness specifically flips ~50% of the time across the test set — the most volatile of 10 measured dimensions. Most other judgments are more stable; the average vision–text flip rate is 16–22%."

Previously the caption read "Same sentence • Same AI model • Identical text content • Different font • Opposite verdict," which implied the flip is universal. The new caption keeps the visual punch but tells the reader explicitly that trustworthiness is the cherry-picked dimension and that most judgments are more stable.

### 5. Decision Asymmetry Spotlight callout (NEW)

A visually prominent blue-bordered callout was inserted in `index.html` immediately after the Key Findings Summary table, before "Why This Might Matter." In `index.md` it appears as a blockquote in the same position. Content: a one-sentence headline ("Borderline triage cases are less likely to reach a human reviewer when processed by vision pipelines, with direction varying by provider"), the 73% / 100% / neutral breakdown, an example of where this matters in practice, and a note that sample size is modest.

This is the takeaway most readers should remember, surfaced near the top of the findings.

A new CSS class `.spotlight-box` was added in the `<style>` block for this.

### 6. Findings re-sequenced

| New # | Title | Source |
|------|-------|--------|
| 1 | Vision pipelines diverge from text pipelines | unchanged |
| 2 | **Decision asymmetry — vision pipelines deflect away from escalation** | **PROMOTED from old #6** |
| 3 | Typography modulates the divergence (up to ~5pp) | DEMOTED from old #2; title bound with "~5pp" |
| 4 | Some dimensions are much more affected than others | shifted from old #3; added a sentence calling out that the "~50%" trustworthiness number is dimension-specific |
| 5 | When disagreements occur, they have direction | shifted from old #4 |
| 6 | Font choice also has directional effects | shifted from old #5 |
| 7 | Some dimension flips strongly predict decision flips | **PROMOTED from "Which dimensions predict decision flips?" sub-section to a numbered finding**; added a sentence noting that trustworthiness, despite being most volatile, is *not* a strong predictor of decision flips — explaining why the ~50% headline overstates practical stakes |

Anchor IDs `#finding-1` … `#finding-7` were added in the HTML so the spotlight callout and Related Work section can link to specific findings. `#limitations` was also added.

### 7. Related Work (NEW section)

Added between "What I Found" and "Limitations" in both files. Four short paragraphs:

1. **Prompt-format sensitivity** — positions against Sclar et al. (text-mode formatting sensitivity) as the analog in vision-mode.
2. **VLM hallucination and evaluation** — positions as complementary to object-level hallucination benchmarks (we ask whether judgments on extracted content are stable, not whether extraction is correct).
3. **Visual / typographic robustness and prompt injection** — distinguishes from adversarial typography work (these fonts are non-adversarial; focus is on judgment shifts not extraction).
4. **What's new here** — three-bullet summary of the contribution.

All citations are placeholders that **need Will's input** — see next section.

### 8. Bounded numbers throughout

Wherever "20–50%" or "~50%" or "16–22%" appeared, the surrounding prose now clarifies what subset of the data the number describes (average across dimensions vs. trustworthiness specifically vs. typography modulation). The catchy framings are preserved but contextualized.

## Judgment calls on ambiguous framing

A few decisions where the original task allowed latitude:

1. **Headline wording** — Used "Vision pipelines deflect borderline cases away from human review — and typography modulates the bias" verbatim from the suggested direction in the task. "Modulates" feels slightly more academic than the existing voice, but the sense is closer to "modifies / influences" than "causes," which is the point.
2. **Decision Asymmetry Spotlight placement** — Put it after the Key Findings Summary table and before "Why This Might Matter." This matches the spirit of "near the top of findings" without burying the existing summary table.
3. **Spotlight styling** — Used a blue-bordered box (left border accent) rather than the yellow `.why-matter-box` style, so the two callouts are visually distinct. CSS class is `.spotlight-box` in the `<style>` block.
4. **Related Work placement** — Put it after "What I Found" and before "Limitations." Other reasonable options (e.g., before "What I Did") were considered but I chose this because the section makes more sense once the reader has seen what was actually measured.
5. **Anthropic claim in spotlight** — The Anthropic decision flip rate is 4.9% with 0% directional bias, which I described as "roughly neutral." This is stronger than "no signal" and weaker than "significantly different from OpenAI/Google" — the original task framing as "73–100% depending on provider" technically excludes Anthropic, so I made the asymmetry text say explicitly "73% (OpenAI), 100% (Google), and roughly neutral (Anthropic)" to avoid implying universality.
6. **Finding 7 framing** — Promoted "Which dimensions predict decision flips?" to a numbered finding and added a sentence pointing out that trustworthiness (most volatile dimension) is *not* a strong predictor of decision flips. This is in the existing data (`trustworthy: 0.7×` lift) and reinforces the point that the headline ~50% trustworthiness number overstates practical stakes. I considered this an editorial highlight rather than a new finding.
7. **TL;DR as ordered list** — The original TL;DR was a single bold paragraph. I converted it to an ordered list of three numbered findings because the "three nested findings" structure is hard to read in a single sentence. If Will prefers prose, easy to revert.
8. **MD vs. HTML parity** — The two files were kept in sync at the content level. The MD file uses a blockquote for the Decision Asymmetry Spotlight; the HTML uses the styled `.spotlight-box` div.

## Citation placeholders that need Will's input

All inside the **Related Work** section in both `docs/index.html` and `docs/index.md`:

1. **Sclar et al. on prompt-format sensitivity.** Tagged: `[CITATION NEEDED: Sclar et al., "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design" / ICLR 2024 — please confirm exact title, authors, and venue]`. Best guess at title/venue from memory; confirm before publishing.
2. **VLM hallucination / robustness benchmarks.** Tagged: `[CITATION NEEDED: representative VLM hallucination / robustness benchmarks — e.g., POPE, MM-Vet, MMHal-Bench, HallusionBench — please select and cite the ones you want to anchor against]`. I named candidates but did not pick which ones to actually cite, since the choice is a positioning decision.
3. **Visual prompt injection / typographic adversarial examples.** Tagged: `[CITATION NEEDED: representative work on visual prompt injection and typographic adversarial examples for VLMs — please pick the ones most relevant to the framing here]`. No specific paper named — the literature is fast-moving and I don't want to hallucinate.

## Things deliberately NOT changed

Per the task's "What NOT to change" list:

- All data tables, numerical results, CIs, lift values, percentages — unchanged
- "What I Did" / methodology section — unchanged (one minor copy edit was avoided)
- "Limitations" section — unchanged (already honest)
- "What's Next" section — unchanged
- Citation block (BibTeX) — unchanged
- Interactive demo data (`VARIANTS`, `PROVIDERS`, `DEMO_SENTENCE`) and JS — unchanged
- Figure references — all 7 distinct figures verified to still exist on disk
- "Why This Might Matter" callout (already moved up in the file in a prior pass) — kept in place; content unchanged

## Build / preview check

There is no Jekyll, Eleventy, or Hugo build system in this repo. `docs/index.html` is served directly by GitHub Pages. Verification done:

- HTML tag balance check via Python `html.parser` — `unclosed: none`. (The parser surfaces spurious "errors" on `<meta ... />` self-closing tags — these are valid HTML5, not real errors.)
- All 7 referenced figures exist on disk under `docs/figures/`.
- All internal anchors (`#findings`, `#demo`, `#finding-1` … `#finding-7`, `#limitations`) cross-check against their targets.
- Interactive demo JS data block is byte-identical to the original.

I did not open the page in a browser since this was an automated revision pass. If anything looks visually wrong after publishing, the most likely suspects are the new `.spotlight-box` CSS (mobile narrow-screen padding) and the ordered-list TL;DR in the `.tldr-box` (inline `style` may want to move to the stylesheet).

## Broken / unresolved

None. All references and anchors check out.

## Suggested follow-ups (out of scope for this pass)

- Fill in the three Related Work citations.
- If you keep this framing, consider regenerating the OG/Twitter card image to feature the decision-asymmetry chart explicitly with a one-line caption (the meta tag now points at `comparison_decision_flip.png` but a designed social card would land harder).
- The `tldr-box` ordered list uses inline styling — fine for a static page, but tidier to move to the stylesheet if you do another design pass.
