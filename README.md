# ThinkingType

A lightweight benchmark measuring how typography and presentation influence
vision-language-model judgments and decisions. The same content is shown to the
same model as plain text and as a rendered image (varying font, size, layout,
and color), and ThinkingType measures how often the answer changes. Results,
interactive demos, and the full writeup: **https://wdanfort.github.io/thinkingtype/**

The benchmark has two tracks:

- **Sentence track (judgments):** 120 synthetic sentences x 8 fonts x 10 yes/no
  questions (urgent? trustworthy? professional?). Metric: flip rate — how often
  the model's image answer disagrees with its own text answer.
- **Decision track (gates):** ~330 synthetic decision items across three tests
  (moderation removal, resume screening, hardship-grant appeal), calibrated to
  each model's own decision boundary. Metric: decision shift — how much more
  often the model says yes to the image than to the identical text.

The full fact sheet of results lives in [FACTS.md](FACTS.md).

## Installation

```bash
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

**Tesseract OCR** is required for the sentence track's legibility-check stage:

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

Bundled fonts (all open-source licensed):
- Liberation Serif (Regular, Bold)
- Liberation Sans (Regular, Bold)
- Liberation Mono (Regular)
- Comic Neue (Regular) — an open-source Comic Sans analogue; site prose says
  "Comic Sans" as shorthand for this font
- OpenDyslexic (Regular)

## API keys

Set environment variables or use a `.env` file:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

## Reproducing the v1 results

### Decision track (gates)

Matches the site's Reproduce section:

```bash
typo-eval --config configs/gates_v1.yaml gates-build
typo-eval --config configs/gates_v1.yaml gates-render
typo-eval --config configs/gates_v1.yaml gates-calibrate --provider <p>
typo-eval --config configs/gates_v1.yaml gates-run --provider <p>
typo-eval --config configs/gates_v1.yaml gates-analyze
```

Follow-up experiment configs: `gates_v1_rubric.yaml` (criteria written into the
prompt), `gates_v1_realism.yaml` (forum/phone screenshot chrome),
`gates_v1_fair.yaml` (OpenDyslexic fairness scale-up), and `gates_v2.yaml`
(July 2026 release-tracking rerun). Compare any two runs with:

```bash
typo-eval gates-drift --run-a gates_v1 --run-b gates_v2
```

Per-run reports, item-level CSVs, and forest plots are in
[results/gates/](results/gates/). The drift monitor for testing your own
model x gate combinations is documented in [docs/GATE_DRIFT.md](docs/GATE_DRIFT.md).

### Sentence track (judgments)

```bash
typo-eval --config configs/v0c_release.yaml generate
typo-eval --config configs/v0c_release.yaml render
typo-eval --config configs/v0c_release.yaml ocr
typo-eval --config configs/v0c_release.yaml run --provider <p>
typo-eval --config configs/v0c_release.yaml analyze
```

`v0c_release.yaml` is the July 2026 administration (Claude Fable 5, GPT-5.6
Sol); `v0b_frontier.yaml` and `v0b_oldgen_fixed.yaml` are the earlier
administrations (GPT-5.5 / Claude Sonnet 5, and GPT-4o). All use seed 42 and
identical variants, so every administration judges the same 120 sentences.

**Text baseline note:** the text arm sends the original sentence text directly
to the model (`data/inputs/sentences.csv`); it is not derived from OCR. The
`ocr` stage runs Tesseract on the rendered images as a separate legibility
check only — its output is not part of the text-vs-image comparison, so OCR
errors cannot confound the flip rates.

## Models used in the v1 results

Models are configurable per provider in each YAML config (`model_text` /
`model_vision`). The published v1 numbers come from:

| Track | Models |
|---|---|
| Sentence (judgments) | GPT-4o, GPT-5.5, Claude Sonnet 5, Claude Fable 5, GPT-5.6 Sol |
| Decision (gates) | GPT-5.5, Claude Sonnet 5, Gemini 3.5 Flash, Claude Fable 5, GPT-5.6 Sol |

The sentence track was not run on Gemini (the decision track was).

## CLI reference

```bash
typo-eval --config <config.yaml> <command>
```

Sentence track: `generate`, `render`, `ocr`, `run [--provider ...] [--limit N]
[--shard i/N]`, `analyze [--run <run_id>]`, `compare`, or `all` for the full
pipeline.

Decision track: `gates-build`, `gates-render`, `gates-calibrate --provider <p>`,
`gates-run --provider <p>`, `gates-analyze`, and
`gates-drift --run-a <A> --run-b <B> [--provider <p>]`.

## Directory structure

```
configs/            # YAML configs (v0*/v0c_* sentence track, gates_* decision track)
assets/fonts/       # Bundled open-source fonts
data/
  inputs/           # Sentence CSV + gate item banks (committed)
  rendered/         # Rendered stimulus images (generated locally)
  ocr/              # Tesseract output (legibility check)
  manifests/        # Run metadata
results/
  gates/            # Decision-track reports, analysis CSVs, drift reports (committed)
  runs/<run_id>/    # Raw JSONL response logs + per-run analysis (generated locally)
src/typo_eval/
  cli.py            # CLI entrypoint
  config.py         # Configuration schema
  render.py         # Image rendering
  ocr.py            # Tesseract OCR
  inference.py      # Model inference
  analysis.py       # Sentence-track analysis
  gates/            # Decision-track pipeline (stimuli, calibration, analysis, drift)
  providers/        # OpenAI, Anthropic, Google
scripts/
  fetch_fonts.py
  install_tesseract.sh
tests/
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff format src/ tests/
ruff check src/ tests/ --fix
```

## License

MIT License. See [LICENSE](LICENSE). Fonts are licensed under their respective
open-source licenses (OFL, GPL+exception).
