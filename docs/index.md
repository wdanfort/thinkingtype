# t | h | i | n | k | i | n | g | t | y | p | e

## Vision-language models "read" text differently than LLMs — and frontier models do it about half as much.

A small, controlled test of how visual presentation changes the way vision-language models (VLMs) judge text, and whether newer models have closed the gap.

*Independent research on my own time; views are my own, not my employer's.*

**TL;DR.** I gave three models the same sentences as plain text and as images of that text, then asked ten yes/no questions about each. The answers disagreed 8–15% of the time. Two of the disagreements point the same way in every model I tested: rendered text reads as less formal and less professional than the identical words given as text. The gap shrank by about half between GPT-4o (15.2%) and the current frontier models (GPT-5.5 at 7.8%, Claude Sonnet 5 at 9.5%). Font choice moves it a few points. The model and its generation matter more.

## Motivation

I read *Thinking with Type* and got interested in how font, weight, and emphasis change the way people read the same words. More and more systems now have VLMs read documents directly (resumes, forms, receipts, screenshots), so I wanted to know whether the same typographic cues move a model's judgments when the underlying text is identical. If they do, typography is an unspoken input to any pipeline that lets a VLM look at a page.

## What I did

I built a small harness that separates presentation from content:

- **120 synthetic sentences** across 6 everyday categories (neutral, calls-to-action, authority, warnings, promotional, procedural), written to sit near the border between categories.
- **8 typography variants** (Times regular/bold, Arial regular/bold, Arial caps-only at regular weight, Monospace, Comic Sans, OpenDyslexic) plus a plain-text baseline.
- **10 yes/no judgment dimensions** (trustworthy, professional, formal, urgent, persuasive, confident, high-risk, and so on).
- **3 models across two generations**: GPT-4o (previous generation), GPT-5.5 and Claude Sonnet 5 (current frontier). The same model reads both the text and the image, so the comparison is text versus image, not model versus model.

A **flip** is when the image answer differs from the text answer on the same sentence and question. Every reported number carries a 95% confidence interval from a bootstrap that resamples by sentence (the unit of independence), and directional claims are checked with an exact McNemar test with Benjamini–Hochberg correction across the ~30 comparisons.

Built with Claude Code. → [Code and data](https://github.com/wdanfort/thinkingtype)

## What I found

### 1. The text/image gap is real, and frontier models cut it roughly in half

Between 8% and 15% of yes/no judgments change when the input switches from text to an image of the same text. The gap is widest on the older model and smaller on both frontier models; the confidence intervals don't overlap.

![Text/vision gap by model generation](figures/v0b_generation_gap.png)

| Model | Generation | Flip rate | 95% CI |
|-------|-----------|----------:|--------|
| GPT-4o | previous | 15.2% | [13.5%, 17.1%] |
| Claude Sonnet 5 | frontier | 9.5% | [8.1%, 10.9%] |
| GPT-5.5 | frontier | 7.8% | [6.7%, 8.9%] |

This gap exists before font choice enters the picture; it comes from the switch in modality alone. That it shrank by generation is the most encouraging result here. Whatever causes it, newer models have less of it.

### 2. Two disagreements are consistent across models; the rest aren't

When text and image judgments disagree, do they disagree in a consistent direction? For two dimensions, yes, in all three models:

- **Formal**: rendered text reads as less formal than the same words as text (net −53% to −83%).
- **Professional**: rendered text reads as less professional (net −74% to −100%).

The rest depend on the model. "Persuasive" swings +99% (more persuasive) on GPT-4o but −22% on GPT-5.5; "confident," "urgent," and "high-risk" all flip sign across providers. There is no single direction VLMs push in. Different models have learned different associations with how text looks on the page.

![Direction of disagreement by dimension and model](figures/v0b_directionality.png)

### 3. Some dimensions move far more than others

The headline 8–15% is an average across ten dimensions, and per dimension it spreads out. On GPT-4o, "confident" (35.5%) and "trustworthy" (27.0%) flip often, while "emotional" (1.9%) and "form-dependent" (0.2%) barely move. Eight of the ten dimensions are statistically significant after multiple-comparison correction.

![Flip rate by dimension](figures/v0b_by_dimension.png)

### 4. Font choice moves the gap a few points

Within the image condition the font matters, though less than the modality switch itself. Standard serif and sans faces (Times, Arial regular) are the most stable in every model. The widest-swinging variant differs by model: OpenDyslexic on GPT-4o (21.1%), Comic Sans on GPT-5.5 (14.1%), caps-only on Sonnet 5 (10.9%). (The caps variant is set at regular weight, so capitalization is tested separately from boldness.)

![Flip rate by typography variant](figures/v0b_by_variant.png)

### 5. The "send it to a human?" decision barely moves

The question I most wanted to answer: when a model triages content, does reading it as an image instead of text change whether it flags the case for a human? I asked each model "Should this be escalated to a human reviewer for safety, compliance, or potential user harm?" on the same text-versus-image basis.

The decision turned out to be robust to the modality switch, but I also couldn't build a clean test of the case that matters. The decision only moves near its threshold, on content the model is torn about, and both of my stimulus sets landed far from it. On the 120 everyday sentences, the text models escalate almost nothing (7, 1, and 0 of 120 for GPT-4o, GPT-5.5, and Sonnet 5). So I wrote 48 sentences with real cause for concern (credential requests, threats, medical advice, privacy disclosures), and the text models escalate nearly all of them (44–45 of 48).

With the baseline pinned to one extreme or the other, the few flips that remain mostly reflect whichever direction was still available. Across the three models on the concerning set the direction isn't consistent (GPT-4o net +100%, GPT-5.5 −42%, Sonnet 5 +33%), and only GPT-4o's reaches significance (McNemar p=0.001; the others p=0.06 and p=0.39).

One thing I want to follow up on: GPT-5.5 escalated fewer account-security alerts as images than as text. Nine of its 17 dropped escalations were messages like "Unusual sign-in activity was detected from a new device," which it flagged as text but not as an image. A triage pipeline would care about exactly that kind of quiet drop, but at p=0.06 it needs a bigger sample before I'd call it a finding.

The strong version of the claim I started with, that VLMs quietly drop borderline cases, did not survive a same-model test with adequate power. Testing it properly needs escalation content calibrated to the decision boundary, which is the main open problem this project surfaced.

## Does this change any decisions?

Most of what I measured is a shift in subjective judgments: formal, professional, trustworthy. A flip on those isn't an error, since there is no correct answer, and by itself it doesn't tell anyone to do anything differently. It starts to matter when the judgment feeds a decision someone owns (approve or deny, escalate or don't, pass or fail) and the typography pushes a borderline case over the line. The escalation experiment was supposed to test that link and couldn't (Finding 5). For now, these results describe an input to such decisions rather than a measured effect on them.

Where I expect it to show up:

**Product teams running VLM triage or scoring.** If a VLM reads documents and something downstream keys on its tone or quality judgment (triage, ranking, auto-approval), typography is an input you aren't controlling. The effect is shrinking with model generation, but "less formal and less professional when rendered" held in every model I tested. If you score professionalism from a page image, run your own gate on the same content in two fonts before trusting it.

**Accessibility and fairness.** Unlike the rest, this one doesn't come down to taste. If models judge accessibility-formatted documents (OpenDyslexic) differently on identical content, that's a fairness problem with legal weight. The effect I saw is small and needs more evidence, but of everything here it's the result most likely to require action, and the one I'd pin down next.

**Robustness research.** The text/image gap on identical content is an inconsistency worth tracking over time. Since its direction varies by model, it looks learned rather than architectural.

## Related work

**Visual text style and VLM judgments.** The closest precedent is Wang, Larson, and Zhao ([2026](https://arxiv.org/abs/2604.27553)), who render a concept word in two style families and show the VLM's attribute description shifts with visual style even when the concept is read correctly, which they call style leakage from surface into semantics. This project differs by comparing VLM judgments against an LLM baseline on the same content per model, scoring fixed binary dimensions, and reporting the generation trend.

**The typography gap in VLM perception.** Zhou et al. ([*Reading ≠ Seeing*, 2026](https://arxiv.org/abs/2603.08497)) build FontBench and find VLMs transcribe text near-perfectly but judge typographic properties (font family, style) poorly. The visual surface is processed unevenly, which is consistent with the modality gap measured here.

**Text bias in multimodal models.** "Text Speaks Louder than Vision" ([2025](https://arxiv.org/abs/2504.01589)) shows VLMs lean on textual and linguistic priors over visual evidence. This project comes at the same tension from the other side: here the rendering of identical text perturbs the judgment.

**Prompt-format sensitivity.** Sclar et al. ([2024](https://arxiv.org/abs/2310.11324)) show LLM accuracy swings across equivalent prompt formats; this is the visual analogue.

**Typographic attacks.** Goh et al. ([Distill 2021](https://distill.pub/2021/multimodal-neurons/)) and Qi et al. ([AAAI 2024](https://arxiv.org/abs/2306.13213)) show adversarial text and visuals can steer multimodal models. The fonts here are ordinary, and the models read them correctly; what I'm measuring is judgment drift downstream of correct reading.

## Limitations

- **Sentence-level, synthetic.** Short synthetic sentences, not real documents. Effects may shrink or compound at document scale.
- **Rendering confounds.** Fonts differ in ink coverage and rendered width even at matched point size (I record image size and ink ratio per variant). Some of the font effect is these low-level differences rather than style.
- **Reproducibility.** Temperature 0 plus an API seed on OpenAI. The frontier models accept neither a temperature nor a seed, so their determinism is best-effort.
- **No mechanism.** I can describe the patterns, not explain why a given model reads a given font the way it does.

## Future directions

**Test the decision link.** The open question is whether any of this moves a decision rather than just a judgment. The test I'd run: pick a real gate (résumé screening, moderation triage, eligibility review), write cases the text model decides at roughly 50/50, and measure whether rendering the same content as an image changes the outcome, and in which direction. Both of my stimulus sets missed that boundary, in opposite directions (Finding 5). If there's an effect, that's a concrete risk for VLM pipelines. If there isn't, that's also useful: it means the gate is robust and images are safe to feed it. Either answer changes what someone builds.

**Pin down the accessibility effect.** Does the OpenDyslexic gap hold at larger sample sizes, does it lean harsher, and does it move a real gate? If yes, that's a compliance question for anyone scoring documents with a VLM.

**Connect judgments to decisions.** Does "reads as less professional" predict a hire or approve flip? Cheap to measure once the gate above exists, and it's the mechanism that would make the subjective shifts matter.

**Real documents, and beyond fonts.** Résumés, intake notes, benefits appeals; then color, highlighting, and layout. Typography is one visual cue among many.

**Track the gap over time.** It halved between generations. This is the first datapoint in watching whether it keeps closing.

## Citation

```bibtex
@misc{danforth2026thinkingtype,
  author = {Danforth, Will},
  title  = {thinkingtype: Vision-language models "read" text differently than LLMs},
  year   = {2026},
  url    = {https://github.com/wdanfort/thinkingtype}
}
```
