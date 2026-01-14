# Typography Evaluation Harness v0

A reproducible CLI pipeline for evaluating how AI vision models interpret typographic presentation of text. This tool generates typography-rendered images, runs OCR baselines, and measures how visual presentation affects model judgments.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the full pipeline (requires API keys)
typo-eval --config configs/v0_default.yaml all

# Or run individual steps
typo-eval generate --config configs/v0_default.yaml
typo-eval render --config configs/v0_default.yaml
typo-eval ocr --config configs/v0_default.yaml
typo-eval run --config configs/v0_default.yaml --provider openai
typo-eval analyze --config configs/v0_default.yaml
```

## Installation

### Python Dependencies

```bash
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### System Dependencies

**Tesseract OCR** is required for OCR baseline generation:

```bash
# Linux (Debian/Ubuntu)
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Or use the provided script
./scripts/install_tesseract.sh
```

### Fonts

Open-source fonts are bundled in `assets/fonts/`. To download fresh copies:

```bash
python scripts/fetch_fonts.py
```

Required fonts (all OFL/open-source licensed):
- Liberation Serif (Regular, Bold) - Serif
- Liberation Sans (Regular, Bold) - Sans-serif
- Liberation Mono (Regular) - Monospace
- Comic Neue (Regular) - Comic style
- OpenDyslexic (Regular) - Accessibility

## Configuration

Configuration is via YAML files in `configs/`:

- `v0_default.yaml` - Full pipeline with sentences enabled
- `v0_sentences_only.yaml` - Sentences only
- `v0_artifacts_only.yaml` - Artifacts only

### Key Configuration Options

```yaml
seed: 42  # For reproducibility

inputs:
  sentences:
    enabled: true
    n_sentences: 36
  artifacts:
    enabled: false

inference:
  mode: both  # "dimensions", "decision", or "both"
  temperature: 0.0
  dimensions:
    - urgent
    - formal
    - trustworthy
    # ... 10 total

providers:
  openai:
    model_text: gpt-4o-mini
    model_vision: gpt-4o
```

## Pipeline Phases

### 1. Generate (`typo-eval generate`)

Creates input datasets:
- `data/inputs/sentences.csv` - Sentence texts with categories
- `data/inputs/artifacts.csv` - Artifact texts (if enabled)

### 2. Render (`typo-eval render`)

Renders typography images:
- `data/rendered/sentences/sentence_###/<variant>.png`
- Uses container framing to prevent clipping

### 3. OCR (`typo-eval ocr`)

Runs Tesseract OCR on rendered images:
- `data/ocr/sentences/sentence_###.txt` - Deduped OCR baseline per sentence
- OCR is run once per sentence (not per variant) for proper comparison

### 4. Run (`typo-eval run`)

Runs inference with configurable modes:

**Dimensions Mode**: 10 binary perception questions (urgent, formal, trustworthy, etc.)

**Decision Mode**: Escalation judgment ("Should this be escalated to a human reviewer?")

Outputs:
- `results/runs/<run_id>/raw/responses.jsonl` - Canonical append-only log
- `results/runs/<run_id>/raw/responses.csv` - Derived table

Supports resume: rerunning skips completed calls automatically.

### 5. Analyze (`typo-eval analyze`)

Computes:
- Flip rates vs OCR baseline (by variant, sentence, dimension)
- Bootstrap confidence intervals
- Heatmaps (sentence x variant)

Outputs:
- `results/runs/<run_id>/analysis/summary.md`
- `results/runs/<run_id>/analysis/flip_*.csv`
- `results/runs/<run_id>/analysis/figures/*.png`

## Directory Structure

```
configs/
  v0_default.yaml
assets/
  fonts/           # Bundled open-source fonts
data/
  inputs/          # Generated sentences/artifacts
  rendered/        # Typography images
  ocr/             # OCR text baselines
  manifests/       # Run metadata
results/
  runs/<run_id>/
    raw/           # responses.jsonl, responses.csv
    analysis/      # CSVs, figures, summary.md
src/typo_eval/
  cli.py           # CLI entrypoint
  config.py        # Configuration schema
  render.py        # Image rendering
  ocr.py           # Tesseract OCR
  inference.py     # Model inference
  analysis.py      # Analysis and plotting
  providers/       # OpenAI, Anthropic, Google
scripts/
  fetch_fonts.py
  install_tesseract.sh
tests/
```

## CLI Reference

```bash
# Full pipeline
typo-eval --config <config.yaml> all

# Individual commands
typo-eval --config <config.yaml> generate
typo-eval --config <config.yaml> render
typo-eval --config <config.yaml> ocr
typo-eval --config <config.yaml> run [--provider openai|anthropic|google] [--dry-run] [--limit N]
typo-eval --config <config.yaml> analyze [--run <run_id>]
```

### Options

- `--config` - Path to YAML config file
- `--provider` - Inference provider (openai, anthropic, google)
- `--dry-run` - Print planned calls without executing
- `--limit N` - Limit number of inference calls
- `--run <run_id>` - Specify run ID for analysis

## API Keys

Set environment variables or use `.env` file:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

## Key Design Decisions

1. **Deduped OCR**: OCR baseline is run once per sentence (not per variant) to avoid inflating comparison counts.

2. **Container Framing**: Images use rounded rectangle containers to prevent the "poster effect" where edge rendering artifacts affect model perception.

3. **Resume Support**: JSONL logging with deduplication allows interrupted runs to resume without re-running completed calls.

4. **Strict Parsing**: Yes/No responses are parsed strictly with clear error handling for invalid responses.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
ruff format src/ tests/
ruff check src/ tests/ --fix
```

## License

MIT License. See LICENSE file.

Fonts are licensed under their respective open-source licenses (OFL, GPL+exception).
