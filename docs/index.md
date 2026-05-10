# t | h | i | n | k | i | n | g | t | y | p | e

## Vision pipelines deflect borderline cases — and typography modulates the bias

A controlled diagnostic evaluation of how visual presentation influences interpretation in vision-language models.

Disclaimer: This is independent research conducted on my own time. The views expressed here are my own and do not represent those of my employer.

**TL;DR — three nested findings:**

1. **Vision–text divergence (baseline).** Across three frontier providers, 16–22% of binary judgments flip between vision and text-only pipelines on identical content — before any typography variation.
2. **Decision asymmetry (the practical concern).** When the disagreement involves a triage decision ("escalate to a human reviewer?"), 73–100% of flips point toward "don't escalate" for OpenAI and Google; Anthropic is roughly neutral. Borderline cases are systematically less likely to reach a human when processed visually.
3. **Typography modulation (the colorful angle).** Font choice modulates the divergence by up to ~5pp on the flip rate. Trustworthiness — the most volatile of the 10 dimensions measured — flips ~50% of the time; most dimensions are more stable.

## Motivation

I recently read *Thinking with Type* and was struck by how text font, weight, and emphasis shape how humans interpret the same words (and how these choices are shaped by the ideas and technology of the day). As more systems rely on vision-language models to read documents, I wanted to understand how those same typographic signals are interpreted: do they change judgments when the underlying text is identical? How should we think about burgeoning vulnerabilities and opportunities for UI/UX? This project is a small, controlled evaluation designed to explore the vision-text model divergence and implications for product design. 

## What I Did

I built a controlled diagnostic evaluation harness to isolate presentation effects:

- **36 synthetic sentences** across 6 semantic categories (neutral, authority, warnings, calls-to-action, promotional, procedural)
- **8 typography variants** (Times regular/bold, Arial regular/bold/ALL CAPS, Comic Sans, Monospace, OpenDyslexic)
- **3 providers** (OpenAI gpt-4o, Anthropic claude-sonnet-4, Google gemini-2.0-flash)
- **10 binary interpretation dimensions** (trustworthy, professional, formal, urgent, persuasive, etc.)
- **1 downstream decision task** ("Should this be escalated to a human reviewer?")
- **Temperature = 0.0** for reproducibility (I also ran this at 0.3 for robustness)

A **flip** = the vision model's YES/NO answer differs from the text model's answer on identical text content.

**Total comparisons:** ~9,000 dimension judgments + ~1,000 decision judgments across providers

This is sentence-level only. I have not yet tested whether these patterns hold for full documents. Development executed almost exclusively with Claude Code. 

→ [Code and results](https://github.com/wdanfort/thinkingtype) | 


## What I Found (V0 - 1/21/26)


### Key Findings Summary

| Provider | Model | Dimension Flip Rate | Decision Flip Rate | Decision Direction | Most Variable | Most Stable |
|----------|-------|:-------------------:|:------------------:|:-------------------|:--------------|:------------|
| OpenAI | gpt-4o | 22.2% [19.0%, 25.1%] | 20.8% [10.4%, 31.0%] | -73.3% (Less likely to escalate) | OpenDyslexic (28.4%) | Times Bold (19.7%) |
| Anthropic | claude-sonnet-4 | 15.7% [13.4%, 18.1%] | 4.9% [0.0%, 8.3%] | +0.0% (Mixed) | Arial ALL CAPS (23.6%) | OpenDyslexic (13.9%) |
| Google | gemini-2.0-flash | 16.5% [14.8%, 18.2%] | 11.1% [4.2%, 19.0%] | -100.0% (Less likely to escalate) | Comic Sans (21.9%) | Monospace (14.4%) |

*Flip rates show vision-text disagreement. Decision direction shows net bias when disagreements occur (negative = vision less likely to escalate). Small text variant excluded from analysis.*

> **Decision Asymmetry Spotlight — the takeaway most readers should remember**
>
> Borderline triage cases are less likely to reach a human reviewer when processed by vision pipelines, with direction varying by provider.
>
> When vision and text pipelines disagree on whether to escalate a case for human review, the disagreement is overwhelmingly directional toward *don't escalate*: 73% (OpenAI), 100% (Google), and roughly neutral (Anthropic). For automated triage in medical intake, fraud detection, resumes, or content moderation, this means cases a text-based pipeline would flag may get quietly downgraded by an otherwise-equivalent vision pipeline. Detail in Finding 2 below; sample size is modest — see Limitations.

### 1. Vision pipelines diverge from text pipelines

Processing the same sentence as an **image** instead of **raw text** produces different judgments — even before any typography variation. This is the baseline divergence introduced by visual processing alone, not a typography effect.

Across three providers, **16–22% of binary judgments flipped** between text-only and vision pipelines on identical content (averaged across 10 dimensions; individual dimensions vary widely — see Finding 4).

![Overall flip rate by provider](figures/comparison_overall.png)

| Provider | Model | Flip Rate | 95% CI |
|----------|-------|----------:|--------|
| OpenAI | gpt-4o | 21.8% | [20.3%, 23.4%] |
| Google | gemini-2.0-flash | 16.4% | [15.2%, 17.7%] |
| Anthropic | claude-sonnet-4 | 15.9% | [14.6%, 17.3%] |

### 2. Decision asymmetry — vision pipelines deflect away from escalation

Beyond interpretation dimensions, I tested a downstream **decision task**: "Should this be escalated to a human reviewer?" This is the result with the clearest practical implications, and the one this page is built around.

![Decision flip rate by provider](figures/comparison_decision_flip.png)

| Provider | Decision Flip Rate | Direction |
|----------|-------------------:|-----------|
| OpenAI | 20.8% | 73% toward NO (don't escalate) |
| Google | 11.1% | 100% toward NO |
| Anthropic | 4.9% | Neutral (0% bias) |

**The punchline**: When vision and text pipelines disagree on the escalation decision, OpenAI and Google overwhelmingly fall on the "don't escalate" side; Anthropic is roughly neutral. In practice, borderline cases are less likely to reach a human reviewer when processed visually — with the magnitude depending on the provider.

For systems using vision models to triage documents — medical intake, fraud detection, resumes, or content moderation — cases that a text pipeline would escalate may get quietly downgraded when processed visually. Sample size on the decision task is modest (see Limitations); take the magnitudes as suggestive rather than precise.

### 3. Typography modulates the divergence (up to ~5pp)

Within the vision pipeline, font choice modulates the gap. Some presentations amplify divergence; others keep it near baseline. The effect is real but modest in magnitude — up to about 5 percentage points on the flip rate — which is much smaller than the underlying vision–text divergence in Finding 1.

![Flip rate by typography variant](figures/comparison_by_variant.png)

| Variant | Flip Rate | vs Baseline |
|---------|----------:|-------------|
| Comic Sans | 20.4% | +5pp (amplifies) |
| ALL CAPS | 19.7% | +4pp (amplifies) |
| OpenDyslexic | 19.6% | +4pp (amplifies) |
| Arial Regular | 16.5% | Baseline |
| Times Regular | 15.8% | Baseline |

Standard serif and sans-serif fonts produce the most stable behavior. Stylized fonts (Comic Sans), accessibility fonts (OpenDyslexic), and emphasis treatments (ALL CAPS, bold) increase divergence.

### 4. Some dimensions are much more affected than others

Not all judgments are equally susceptible. Certain semantic dimensions show much higher flip rates (and differ between model providers).

![Flip rate by dimension](figures/comparison_by_dimension.png)

Judgments about **trustworthiness flip roughly half the time** between vision and text pipelines — this is the most volatile dimension and is the source of the "~50%" figure quoted in headline-style framings. Judgments about urgency or emotional tone are much more stable. The 16–22% headline number is the average across all 10 dimensions; the per-dimension picture is more dispersed.

### 5. When disagreements occur, they have direction

When vision and text pipelines disagree, the disagreement is often systematic — but **the direction varies by provider**.

![Direction by provider and dimension](figures/comparison_directionality.png)

| Dimension | OpenAI | Anthropic | Google |
|-----------|--------|-----------|--------|
| trustworthy | Vision → LESS | Vision → LESS | Vision → MORE |
| professional | Vision → LESS | Vision → MORE | Vision → LESS |
| persuasive | Vision → MORE | Vision → MORE | Vision → MORE |
| formal | Vision → LESS | (no flips) | Vision → LESS |
| high_risk | Vision → LESS | Vision → LESS | Neutral |

**What providers agree on:**
- Persuasive: Vision mode makes content seem MORE persuasive (75–93% of flips)
- Formal: Vision mode makes content seem LESS formal (90–100% of flips)

**What providers disagree on:**
- Trustworthy: OpenAI/Anthropic say vision = less trustworthy; Google says more trustworthy
- Professional: OpenAI/Google say vision = less professional; Anthropic says more professional

This inconsistency is itself a finding. You cannot assume that "vision mode" has a universal directional effect — it depends on the model.

### 6. Font choice also has directional effects

Some typography variants systematically push judgments in one direction when flips occur.

![Typography direction bias](figures/direction_bias_net_openai.png)

| Variant | Net Bias | Interpretation |
|---------|----------|----------------|
| Times Bold | -72% (toward NO) | Amplifies negative judgments |
| Arial Regular | -75% (toward NO) | Amplifies negative judgments |
| Times Regular | -67% (toward NO) | Amplifies negative judgments |
| Arial Bold | -68% (toward NO) | Amplifies negative judgments |
| Monospace | -78% (toward NO) | Amplifies negative judgments |
| Comic Sans | -76% (toward NO) | Amplifies negative judgments |
| OpenDyslexic | -78% (toward NO) | Amplifies negative judgments |
| Arial ALL CAPS | -63% (toward NO) | Amplifies negative judgments |

All typography variants show strong negative bias for OpenAI's vision pipeline — when vision and text disagree, vision almost always pushes toward harsher judgments. This is consistent across all font choices, though monospace and OpenDyslexic show the strongest negative bias.

### 7. Some dimension flips strongly predict decision flips

When a dimension judgment flips, how much more likely is the escalation decision to also flip? This connects Findings 4–6 back to the practical concern in Finding 2.

![Dimension-decision lift](figures/dimension_decision_lift_openai.png)

| Dimension | Lift vs Baseline | Interpretation |
|-----------|------------------|----------------|
| urgent | **5.3x** | Strong predictor of decision flips |
| high_risk | 4.6x | Strong predictor of decision flips |
| professional | 3.9x | Strong predictor of decision flips |
| persuasive | 2.4x | Moderate predictor |
| emotional | 1.6x | Moderate predictor |
| formal | 0.6x | Weak/no effect |
| confident | 0.4x | Weak/no effect |
| trustworthy | 0.7x | Weak/no effect |

**Urgent and high_risk judgments are the leading indicators**. When vision mode changes whether something seems "urgent" or "high risk," downstream decisions are ~5x more likely to change too. Professional judgments also strongly predict decision flips (~4x). Notably, trustworthiness — the most volatile dimension — is *not* a strong predictor of decision flips, which is part of why the headline ~50% trustworthiness number can mislead about the practical stakes.

## Why This Might Matter

### For product teams

If your system processes documents visually—resumes, forms, claims, medical records—typography is an implicit input to your model's judgments. The choice between text extraction and vision ingestion is not neutral. **The decision finding is particularly relevant**: vision mode makes borderline cases less likely to be escalated. For triage systems (medical intake, fraud detection, support tickets), this could mean cases that warrant human review get quietly downgraded.

### For people submitting documents reviewed by AI systems (more and more of us!)

Font choice may influence how AI evaluates your content. The effects aren't huge in aggregate, but for borderline cases they could matter. If your document is being processed visually by an AI system, standard fonts (Times, Arial) appear to produce more predictable behavior than stylized alternatives.

OpenDyslexic consistently shows elevated flip rates and a bias toward negative judgments. If AI systems process OpenDyslexic-formatted documents more harshly, that's a potential fairness issue. I'd want more evidence before making strong claims, but it's worth flagging and a thread I want to keep pulling for later iterations.

### For robustness research

Vision–text divergence on identical semantic content is a form of inconsistency worth tracking. The fact that direction varies by provider suggests different models have learned different associations with visual presentation.

## Related Work

This work sits at the intersection of three lines of research and contributes a specific empirical observation rather than a new framework. The goal of this section is to signal awareness of adjacent work, not to be a full lit review.

**Prompt-format sensitivity in language models.** Sclar and colleagues showed that LLM outputs are sensitive to surface-level prompt formatting (whitespace, separators, casing, etc.) on identical underlying tasks — small text-format changes can produce large output swings *[CITATION NEEDED: Sclar et al., "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design" / ICLR 2024 — please confirm exact title, authors, and venue]*. The work here extends that line of inquiry from text-mode formatting to vision-mode typographic formatting: a different surface, but a similar question about robustness to non-semantic presentation.

**Vision-language model hallucination and evaluation.** A growing body of benchmarks documents systematic failure modes in VLMs on tasks where the visual presentation should be irrelevant *[CITATION NEEDED: representative VLM hallucination / robustness benchmarks — e.g., POPE, MM-Vet, MMHal-Bench, HallusionBench — please select and cite the ones you want to anchor against]*. Most of that literature focuses on object-level hallucination or fine-grained recognition. The framing here is complementary: instead of asking whether the VLM correctly extracts content, this eval asks whether *judgments* about the same extracted content are stable across visual presentations.

**Visual / typographic robustness and prompt injection.** Adversarial typography, font-based attacks on OCR, and visual prompt injection are adjacent areas *[CITATION NEEDED: representative work on visual prompt injection and typographic adversarial examples for VLMs — please pick the ones most relevant to the framing here]*. This work differs in two ways: (a) the typographic variations are non-adversarial (standard, naturally occurring fonts), and (b) the focus is on downstream judgment shifts rather than extraction or content-recognition accuracy.

**What's new here.** A small but controlled diagnostic showing that (a) vision–text pipeline divergence on identical content is non-trivial across three frontier providers, (b) the divergence is direction-asymmetric on a triage decision, and (c) typography is a measurable but smaller modulator of (a). This is a starting point, not a final claim — see Limitations.

## Limitations

This is exploratory and generated within limited budget/time constraints. There are also experimental constraints, too: 

- **Sentence-level only** — Effects may attenuate or compound in full documents. 
- **Synthetic content** — Sentences were designed to be ambiguous and at the "border" for the tested categories. 
- **Robustness** — All canonical runs at temperature=0.0, although I also re-run earlier analyses with t=0.3 and found consistent findings. Different prompts may yield different results, too. I plan to stress test this more in a v1 iteration with different prompts, dimension definitions. I am also comparing raw text to images of the text generated. Another confounder might be how I generated, cropped, and standardized image sizes. 
- **No causal mechanism** — I can describe patterns but not explain *why* vision-text differ or why specific fonts drive a specific direction/magnitude.

## What's Next

**Robustnesss**: I plan on testing variation on prompts, dimensionss/decision questions, and a broader range of artifacts to confirm these patterns hold. 

**Realistic documents**: Testing resumes, medical triage notes, and benefits appeals to see if sentence-level patterns transfer to domain-specific decisions.

**Beyond typography**: Color, highlighting, layout structure, UI elements. Typography is one visual signal among many.

**Tracking over time**: Building toward an eval that measures vision–text divergence across model releases.

**Design conventions for AI**: As vision models become ubiquitous—especially with agentic browsing and document processing—there may be value in developing design literacy for AI-facing content to better understand what signals are actually being surfaced. 

---

## Feedback Welcome

This is early. I'd love any feedback or suggestions for future directions. 

- **Code**: [github.com/wdanfort/thinkingtype](https://github.com/wdanfort/thinkingtype)
- **Issues**: [GitHub Issues](https://github.com/wdanfort/thinkingtype/issues)
- **Contact**: wbdanforth [at] gmail [dot] com

## Citation

```bibtex
@misc{danforth2026typography,
  author = {Danforth, Will},
  title = {thinkingtype: Typography Changes How AI Judges Identical Text},
  year = {2026},
  url = {https://github.com/wdanfort/thinkingtype}
}
```

---

*Last updated: January 2026*
