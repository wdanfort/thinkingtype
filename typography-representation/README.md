# Typography, Representation, and AI Interpretation

This project provides a reproducible CLI pipeline that generates typography artifacts, runs text + image inference, and produces analysis tables and plots.

## Quickstart

```bash
cd typography-representation
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
# or use a .env file with OPENAI_API_KEY=your_key_here

# End-to-end run (v0 default)
typography run
```

### Commands

```bash
# Generate artifacts only
typography generate --config configs/v0.yaml

# Run inference only (by config or run id)
typography infer --config configs/v0.yaml
typography infer --run <run_id>

# Analyze a completed run
typography analyze --run <run_id>

# Compare two completed runs
typography analyze --compare <run_id_A> <run_id_B>
```

### Outputs

Runs are stored under `runs/` and artifacts under `artifacts/`:

```
artifacts/<artifact_set_id>/
├── text/<item_id>.txt
├── images/<item_id>/<variant_id>.png
└── metadata.csv

runs/<run_id>/
├── results.csv
├── run.json
├── config_resolved.yaml
├── analysis/
│   ├── tables/
│   ├── figures/
│   └── summary.md
└── logs.txt
```

A registry of runs is maintained in `runs/index.csv`.

## Fonts

The pipeline will try to resolve font paths from font families using the system font registry.
If a font is missing, it will fall back to a known system font and emit a warning in `runs/<run_id>/logs.txt`.
To add custom fonts, place them in `fonts/` and point `variants_v0.csv` to `font_path` entries.

## Adding new models/providers

Implement a new provider in `src/typography/providers/` by subclassing `Provider` and
adding it to the provider registry in `providers/__init__.py`. Providers must implement
text and image inference methods.

## Adding new artifacts

Update `data/sentences_v0.csv` and `data/variants_v0.csv`, then run:

```bash
typography generate --config configs/v0.yaml
```

You can also create new configs in `configs/` for future artifact sets.
