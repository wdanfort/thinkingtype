# Typography Evaluation Research

A systematic study of how typographic presentation affects AI vision models' interpretation and judgment of text content.

## Overview

This research evaluates how different typographic properties influence the behavior of AI vision models (GPT-4 Vision, Claude, Gemini) when processing text. By rendering the same textual content in different fonts, weights, and styles, we measure whether visual presentation meaningfully changes model outputs.

**Research Question**: To what extent do typographic variations affect AI model perception and decision-making?

---

## Key Findings

> 📊 **Findings section** - Add key insights and discoveries from your evaluation runs here.

### Initial Results Summary

| Finding | Impact | Confidence |
|---------|--------|------------|
| (To be populated) | - | - |

---

## Methodology

### Pipeline Overview

The typography evaluation harness consists of five stages:

1. **Generate** - Create input datasets (sentences, artifacts)
2. **Render** - Produce typography variants as images using different fonts
3. **OCR** - Establish baseline with Tesseract OCR
4. **Run** - Execute inference with multiple AI providers
5. **Analyze** - Compute flip rates, confidence intervals, and visualizations

### Evaluation Dimensions

Models are evaluated across:

- **Perception Dimensions**: urgent, formal, trustworthy, friendly, professional, creative, authentic, energetic, masculine, feminine
- **Decision Task**: Escalation judgment ("Should this be escalated to a human reviewer?")

### Font Variants

Tested typographic properties include:

- **Fonts**: Liberation Serif, Liberation Sans, Liberation Mono, Comic Neue, OpenDyslexic
- **Weights**: Regular, Bold
- **Styles**: Normal, Italic (when available)

---

## Results & Analysis

### Detailed Findings

**Coming Soon**: Results from completed evaluation runs will be displayed here with:

- Flip rate matrices (sentence × variant)
- Statistical significance tests
- Bootstrap confidence intervals
- Heatmaps and comparative visualizations

### Data Tables

#### Model Comparison

| Model | Provider | Flip Rate | Confidence |
|-------|----------|-----------|------------|
| (To be populated) | - | - | - |

#### Per-Font Impact

| Font | Variant | Avg Flip Rate | Effect Size |
|------|---------|---------------|-------------|
| (To be populated) | - | - | - |

### Visualizations

**Heatmaps & Charts** - Generated analysis artifacts:
- Flip rate heatmaps (sentences × variants)
- Dimension-wise sensitivity analysis
- Provider comparison plots

---

## Getting Started

To reproduce or extend this research:

### Installation

```bash
# Clone the repository
git clone https://github.com/wdanfort/thinkingtype.git
cd thinkingtype

# Install dependencies
pip install -e .

# Install system dependencies (Tesseract OCR)
sudo apt-get install tesseract-ocr  # Linux/Debian
# or
brew install tesseract               # macOS
```

### Running the Pipeline

```bash
# Full pipeline with default config
typo-eval --config configs/v0_default.yaml all

# Or run individual stages
typo-eval --config configs/v0_default.yaml generate
typo-eval --config configs/v0_default.yaml render
typo-eval --config configs/v0_default.yaml ocr
typo-eval --config configs/v0_default.yaml run --provider openai
typo-eval --config configs/v0_default.yaml analyze
```

### Configuration

Configuration files in `configs/`:
- `v0_default.yaml` - Full pipeline
- `v0_sentences_only.yaml` - Sentences evaluation only
- `v0_artifacts_only.yaml` - Artifacts evaluation only

---

## Technical Details

### Key Design Decisions

1. **Deduped OCR**: OCR baseline run once per sentence (not per variant) to avoid inflating comparison counts

2. **Container Framing**: Images use rounded rectangle containers to prevent the "poster effect" where edge rendering artifacts affect model perception

3. **Resume Support**: JSONL logging with deduplication allows interrupted runs to resume without re-executing completed calls

4. **Multi-Provider Support**: Supports OpenAI (GPT-4 Vision), Anthropic (Claude Vision), and Google (Gemini Vision)

### Output Structure

```
data/
├── inputs/                 # Generated sentences/artifacts
├── rendered/               # Typography-rendered images
└── ocr/                    # OCR text baselines

results/
└── runs/<run_id>/
    ├── raw/
    │   ├── responses.jsonl # Canonical append-only log
    │   └── responses.csv   # Flattened table
    └── analysis/
        ├── summary.md      # Text analysis summary
        ├── flip_*.csv      # Flip rate matrices
        └── figures/        # PNG visualizations
```

---

## References & Links

- [GitHub Repository](https://github.com/wdanfort/thinkingtype)
- [README](../README.md) - Detailed documentation
- [PyProject Configuration](../pyproject.toml)

---

## Citation

If you use this research or evaluation harness in your work, please cite:

```bibtex
@misc{typography-eval,
  title={Typography Evaluation Research},
  author={Typography Team},
  year={2025},
  url={https://github.com/wdanfort/thinkingtype}
}
```

---

## Contact & Contributions

For questions, feedback, or contributions, please:

- Open an [issue](https://github.com/wdanfort/thinkingtype/issues) on GitHub
- Review the [README](../README.md) for development setup

---

**Last Updated**: January 2025 | [Repo](https://github.com/wdanfort/thinkingtype)
