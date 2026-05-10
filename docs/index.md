# t | h | i | n | k | i | n | g | t | y | p | e

## Vision AI flags fewer borderline cases for human review. Font choice changes the size of the gap.

A small, controlled test of how visual presentation changes the way vision-language models read text.

Disclaimer: This is independent research conducted on my own time. The views expressed here are my own and do not represent those of my employer.

**TL;DR, three findings that build on each other:**

1. **Text vs vision on the same content.** Across three providers, 16 to 22 percent of yes/no judgments flip between the text pipeline and the vision pipeline on the same content, before font choice enters the picture.
2. **The practical part.** When the disagreement is about a triage decision ("send this to a human reviewer?"), 73 to 100 percent of decision flips go toward "don't send" for OpenAI and Google. Anthropic is roughly even. Borderline cases are less likely to reach a human when read as an image.
3. **What font choice adds.** Font choice can shift the flip rate by up to about 5 points. Trustworthiness, the most volatile of the 10 dimensions we measured, flips about half the time. Most dimensions are more stable.

## Motivation

I recently read *Thinking with Type* and was struck by how font, weight, and emphasis shape the way humans read the same words (and how those choices are themselves shaped by the ideas and tools of the day). More and more systems use vision-language models to read documents, so I wanted to know: do those same typographic cues change how the models read the same text? What does that mean for design, and for risk? This project is a small, controlled test built to look at the gap between text-mode and vision-mode AI, and what it might mean for product design.

## What I Did

I built a small evaluation harness to isolate the effect of presentation:

- **36 synthetic sentences** across 6 semantic categories (neutral, authority, warnings, calls-to-action, promotional, procedural)
- **8 typography variants** (Times regular/bold, Arial regular/bold/ALL CAPS, Comic Sans, Monospace, OpenDyslexic)
- **3 providers** (OpenAI gpt-4o, Anthropic claude-sonnet-4, Google gemini-2.0-flash)
- **10 yes/no judgment dimensions** (trustworthy, professional, formal, urgent, persuasive, and so on)
- **1 downstream decision question** ("Should this be sent to a human reviewer?")
- **Temperature 0.0** for reproducibility, with a re-run at 0.3 as a check

A **flip** is when the vision model's yes/no answer differs from the text model's answer on the same content.

**Total comparisons:** about 9,000 dimension judgments and about 1,000 decision judgments across providers.

This is sentence-level only. I have not tested whether these patterns hold for full documents. Built almost entirely with Claude Code.

→ [Code and results](https://github.com/wdanfort/thinkingtype) |


## What I Found (V0, 1/21/26)


### Key Findings Summary

| Provider | Model | Dimension Flip Rate | Decision Flip Rate | Decision Direction | Most Variable | Most Stable |
|----------|-------|:-------------------:|:------------------:|:-------------------|:--------------|:------------|
| OpenAI | gpt-4o | 22.2% [19.0%, 25.1%] | 20.8% [10.4%, 31.0%] | -73.3% (less likely to send to a human) | OpenDyslexic (28.4%) | Times Bold (19.7%) |
| Anthropic | claude-sonnet-4 | 15.7% [13.4%, 18.1%] | 4.9% [0.0%, 8.3%] | +0.0% (roughly even) | Arial ALL CAPS (23.6%) | OpenDyslexic (13.9%) |
| Google | gemini-2.0-flash | 16.5% [14.8%, 18.2%] | 11.1% [4.2%, 19.0%] | -100.0% (less likely to send to a human) | Comic Sans (21.9%) | Monospace (14.4%) |

*Flip rates show how often the text pipeline and the vision pipeline disagree. Decision direction shows the net lean when they disagree (negative means vision is less likely to send the case to a human). The small-text variant is left out of the analysis.*

> **The main finding**
>
> Borderline cases are less likely to reach a human reviewer when read as an image. The size of the effect depends on the provider.
>
> When the text pipeline and the vision pipeline disagree on whether to send a case to a human, the disagreement almost always lands on the "don't send" side: 73 percent for OpenAI, 100 percent for Google, and roughly even for Anthropic. For automated triage in medical intake, fraud detection, resumes, or content moderation, that means cases a text-based pipeline would flag may get quietly dropped by an otherwise-similar vision pipeline. Detail in Finding 2 below; sample size is small, see Limitations.

### 1. Text and vision pipelines disagree on the same content

Reading the same sentence as an **image** instead of **plain text** produces different judgments, before font choice enters the picture at all. This is the baseline gap from visual processing alone, not a font effect.

Across three providers, **16 to 22 percent of yes/no judgments flip** between the text pipeline and the vision pipeline on the same content. That is the average across 10 dimensions; some dimensions move much more than others (see Finding 4).

![Overall flip rate by provider](figures/comparison_overall.png)

| Provider | Model | Flip Rate | 95% CI |
|----------|-------|----------:|--------|
| OpenAI | gpt-4o | 21.8% | [20.3%, 23.4%] |
| Google | gemini-2.0-flash | 16.4% | [15.2%, 17.7%] |
| Anthropic | claude-sonnet-4 | 15.9% | [14.6%, 17.3%] |

### 2. When vision and text disagree on a triage decision, vision almost always says "don't send to a human"

Beyond yes/no dimensions, I asked a downstream **decision question**: "Should this be sent to a human reviewer?" This is the finding with the clearest practical implications, and the one this page is built around.

![Decision flip rate by provider](figures/comparison_decision_flip.png)

| Provider | Decision Flip Rate | Direction |
|----------|-------------------:|-----------|
| OpenAI | 20.8% | 73% toward NO (don't send) |
| Google | 11.1% | 100% toward NO |
| Anthropic | 4.9% | Roughly even (0% lean) |

**The punchline:** when the text pipeline and the vision pipeline disagree on whether to send a case to a human, OpenAI and Google almost always land on the "don't send" side. Anthropic is roughly even. In practice, borderline cases are less likely to reach a human when read as an image. The size of the effect depends on the provider.

For systems that use vision models to triage documents (medical intake, fraud detection, resumes, content moderation), cases a text pipeline would send to a human may quietly get dropped when read as an image. Sample size on the decision question is small (see Limitations), so take the exact numbers as suggestive rather than precise.

### 3. Font choice changes the size of the gap (up to about 5 points)

Within the vision pipeline, font choice changes how big the gap with the text pipeline is. Some fonts widen it. Others keep it near the average. The effect is real but small: up to about 5 percentage points on the flip rate, which is much smaller than the text-vs-vision gap itself in Finding 1.

![Flip rate by typography variant](figures/comparison_by_variant.png)

| Variant | Flip Rate | vs Baseline |
|---------|----------:|-------------|
| Comic Sans | 20.4% | +5pp (widens the gap) |
| ALL CAPS | 19.7% | +4pp (widens the gap) |
| OpenDyslexic | 19.6% | +4pp (widens the gap) |
| Arial Regular | 16.5% | Baseline |
| Times Regular | 15.8% | Baseline |

Standard serif and sans-serif fonts produce the most stable behavior. Stylized fonts (Comic Sans), accessibility fonts (OpenDyslexic), and emphasis treatments (ALL CAPS, bold) widen the gap.

### 4. Some dimensions move much more than others

Not all judgments shift equally. A few dimensions flip much more often than the rest, and which dimensions move most differs by provider.

![Flip rate by dimension](figures/comparison_by_dimension.png)

Judgments about **trustworthiness flip about half the time** between text and vision. This is the most volatile of the 10 dimensions, and the source of the "about 50 percent" figure people sometimes quote. Judgments about urgency and emotional tone are much more stable. The 16 to 22 percent headline number is the average across all 10 dimensions, so the picture is more spread out per dimension than the average suggests.

### 5. When the two pipelines disagree, the disagreement has a direction

When text and vision disagree, the disagreement is often systematic, but **the direction depends on the provider**.

![Direction by provider and dimension](figures/comparison_directionality.png)

| Dimension | OpenAI | Anthropic | Google |
|-----------|--------|-----------|--------|
| trustworthy | Vision → LESS | Vision → LESS | Vision → MORE |
| professional | Vision → LESS | Vision → MORE | Vision → LESS |
| persuasive | Vision → MORE | Vision → MORE | Vision → MORE |
| formal | Vision → LESS | (no flips) | Vision → LESS |
| high_risk | Vision → LESS | Vision → LESS | Neutral |

**What providers agree on:**
- Persuasive: vision mode makes content seem MORE persuasive (75 to 93 percent of flips)
- Formal: vision mode makes content seem LESS formal (90 to 100 percent of flips)

**What providers disagree on:**
- Trustworthy: OpenAI and Anthropic say vision = less trustworthy. Google says more trustworthy.
- Professional: OpenAI and Google say vision = less professional. Anthropic says more professional.

That inconsistency is itself a finding. There is no single direction that vision mode pushes in. It depends on the model.

### 6. Font choice also has a direction

Some font variants push judgments in one direction when flips happen.

![Typography direction bias](figures/direction_bias_net_openai.png)

| Variant | Net Lean | Interpretation |
|---------|----------|----------------|
| Times Bold | -72% (toward NO) | Pushes toward harsher judgments |
| Arial Regular | -75% (toward NO) | Pushes toward harsher judgments |
| Times Regular | -67% (toward NO) | Pushes toward harsher judgments |
| Arial Bold | -68% (toward NO) | Pushes toward harsher judgments |
| Monospace | -78% (toward NO) | Pushes toward harsher judgments |
| Comic Sans | -76% (toward NO) | Pushes toward harsher judgments |
| OpenDyslexic | -78% (toward NO) | Pushes toward harsher judgments |
| Arial ALL CAPS | -63% (toward NO) | Pushes toward harsher judgments |

Every font variant shows a strong negative lean for OpenAI's vision pipeline. When text and vision disagree, vision almost always pushes toward harsher judgments. This is true across all font choices, though monospace and OpenDyslexic show the strongest negative lean.

### 7. Some dimension flips strongly predict decision flips

When a dimension judgment flips, how much more likely is the "send to a human?" decision to also flip? This connects Findings 4 through 6 back to the practical part in Finding 2.

![Dimension-decision lift](figures/dimension_decision_lift_openai.png)

| Dimension | Lift vs Baseline | Interpretation |
|-----------|------------------|----------------|
| urgent | **5.3x** | Strong predictor of decision flips |
| high_risk | 4.6x | Strong predictor of decision flips |
| professional | 3.9x | Strong predictor of decision flips |
| persuasive | 2.4x | Moderate predictor |
| emotional | 1.6x | Moderate predictor |
| formal | 0.6x | Weak or no effect |
| confident | 0.4x | Weak or no effect |
| trustworthy | 0.7x | Weak or no effect |

**Urgent and high_risk are the leading indicators.** When vision mode changes whether something looks "urgent" or "high risk," the downstream decision is about 5 times more likely to change too. Professional judgments also strongly predict decision flips (about 4 times). One thing worth flagging: trustworthiness, the most volatile dimension, is *not* a strong predictor of decision flips. That is part of why the headline "about 50 percent" trustworthiness figure can mislead about the practical stakes.

## Why This Might Matter

### For product teams

If your system reads documents as images (resumes, forms, claims, medical records), typography is an unspoken input to the model's judgments. **The decision finding is the one to watch**: vision mode makes borderline cases less likely to be sent to a human. For triage systems (medical intake, fraud detection, support tickets), that means cases that warrant a human look may get quietly dropped.

### For people submitting documents that AI systems read (more and more of us)

Font choice can change how an AI reads your content. The average effect is small, but it can matter for borderline cases. If your document is being read by an AI as an image, standard fonts like Times and Arial seem to produce more predictable behavior than stylized fonts.

OpenDyslexic shows higher flip rates and a lean toward harsher judgments. If AI systems read OpenDyslexic-formatted documents more harshly, that is a possible fairness issue. I want more evidence before making strong claims, but it is worth flagging, and a thread I want to keep pulling on.

### For robustness research

The gap between text-mode and vision-mode AI on the same content is a form of inconsistency worth tracking. The fact that the direction varies by provider suggests different models have learned different associations with how text looks on the page.

## Related Work

This builds on three lines of work. The goal here is to point at the closest neighbors, not to write a full literature review.

**Prompt-format sensitivity in language models.** Sclar and colleagues showed that LLM outputs are sensitive to surface details of prompt formatting (whitespace, separators, casing, and so on) on the same underlying task. Small text-format changes can produce large swings in output *[CITATION NEEDED: Sclar et al., "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design," ICLR 2024. Please confirm exact title, authors, and venue.]*. This project asks the same kind of question for vision-mode AI: a different surface, same kind of robustness question.

**Vision-language model hallucination and evaluation.** A growing set of benchmarks looks at systematic failure modes in VLMs on tasks where the visual presentation should not matter *[CITATION NEEDED: representative VLM hallucination and robustness benchmarks (POPE, MM-Vet, MMHal-Bench, HallusionBench, others). Please pick the ones to anchor against.]*. Most of that work is about whether the VLM gets the contents right (object-level hallucination, fine-grained recognition). The angle here is different: given the contents are read correctly, do the judgments about those contents stay stable when the presentation changes?

**Visual robustness and prompt injection.** Adversarial typography, font-based attacks on OCR, and visual prompt injection are adjacent areas *[CITATION NEEDED: representative work on visual prompt injection and typographic adversarial examples for VLMs. Please pick the ones most relevant to the framing here.]*. This project differs in two ways. First, the fonts here are not adversarial. They are standard, everyday fonts. Second, the focus is on downstream judgment shifts, not on whether the model can extract the text correctly.

**What is new here.** A small, controlled test showing three things: (a) the gap between text-mode and vision-mode AI on the same content is real across three providers, (b) on a triage decision, the gap leans one way (away from sending cases to humans), and (c) font choice changes the size of the gap, but by less than the underlying text-vs-vision gap itself. This is a starting point, not a final claim. See Limitations.

## Limitations

This is exploratory and was built within a small time and budget. A few specific caveats:

- **Sentence-level only.** Effects may shrink or compound in full documents.
- **Synthetic content.** Sentences were written to be ambiguous and to sit at the border of the categories I tested.
- **Robustness.** All main runs were at temperature 0.0. I re-ran earlier analyses at temperature 0.3 and got consistent findings. Different prompts could give different results. I plan to stress test this more in a v1 iteration with different prompts and dimension definitions. I am also comparing raw text to images of that text. How I generated, cropped, and sized those images may be a confounder.
- **No causal mechanism.** I can describe the patterns but not explain *why* text and vision differ, or why specific fonts push in a specific direction.

## What's Next

**Robustness.** I plan to vary the prompts, the dimension and decision questions, and the range of artifacts to see if the patterns hold up.

**Realistic documents.** Testing resumes, medical triage notes, and benefits appeals to see if sentence-level patterns carry over to domain-specific decisions.

**Beyond fonts.** Color, highlighting, layout, UI elements. Typography is one visual cue among many.

**Tracking over time.** Building toward an eval that watches the text-vs-vision gap as models update.

**Design rules for AI-facing content.** Vision models are everywhere now, including in agent browsing and document processing. There may be value in developing design literacy for content that AI is going to read, so we know what signals are actually getting picked up.

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
