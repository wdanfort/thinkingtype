# Typography Eval Pipeline

This folder contains a clean, reproducible, and extensible evaluation pipeline for typography effects in multimodal LLMs. It is derived from the reference notebook export and preserves its behavior while adding explicit schemas, run registry support, and resumable evaluation keyed by run IDs.

## Layout

```
typography_eval/
  config.py
  schemas.md
  data/
    sentences_v1.csv
    variants_v1.csv
    artifacts_v1.csv
  runs/
    runs.csv
    results.csv
  src/
    io_utils.py
    fonts.py
    render.py
    eval.py
    analysis.py
    metrics.py
```

## Dependencies

The core pipeline requires only:

- pandas
- numpy
- tqdm

Rendering and OCR are optional and require Pillow and pytesseract if you run those steps.

## Generate or convert artifacts

### 1) Generate sentences and variants

```python
from typography_eval.src import io_utils

io_utils.generate_sentences_v1()
io_utils.generate_variants_v1()
```

### 2) Convert existing metadata.csv

If you already have `metadata.csv` (from the original notebook export), convert it:

```python
from typography_eval.src import io_utils

io_utils.artifacts_from_metadata()
```

### 3) Or create artifacts from sentences + variants

```python
from typography_eval import config
from typography_eval.src import io_utils

sentences = io_utils.load_sentences()
variants = io_utils.load_variants()
artifacts = io_utils.add_artifacts_from_sentences_and_variants(
    sentences,
    variants,
    config.IMAGES_DIR,
)
io_utils.write_artifacts(artifacts)
```

## Rendering (optional)

Rendering skips PNGs that already exist.

```python
from typography_eval.src import io_utils, render

sentences = io_utils.load_sentences().to_dict("records")
render.render_all(sentences)
```

To render only OpenDyslexic (accessibility):

```python
from typography_eval.src import io_utils, render

sentences = io_utils.load_sentences().to_dict("records")
render.render_accessibility_only(sentences)
```

## Run evaluation

### 1) Register a run

```python
from typography_eval.src import io_utils

run_cfg = io_utils.register_run(
    model_text="gpt-4o-mini",
    model_image="gpt-4o",
    temperature=0.0,
    notes="baseline temp 0.0",
)
```

### 2) Run evaluation with resume support

You must provide your own OpenAI call functions (keeps this repo dependency-light).

```python
from typography_eval.src import eval as eval_mod, io_utils

artifacts = io_utils.load_artifacts()
ocr_cache = eval_mod.load_ocr_cache()

# define call_openai_text(model, temperature, text, question) and
# call_openai_image(model, temperature, image_path, question)

eval_mod.run_evaluation(
    run_id=run_cfg.run_id,
    artifacts=artifacts,
    ocr_cache=ocr_cache,
    call_openai_text=call_openai_text,
    call_openai_image=call_openai_image,
)
```

To run a second temperature (e.g., 0.3), register a new run_id and pass the new temperature:

```python
run_cfg_03 = io_utils.register_run(temperature=0.3, notes="temp 0.3")

# call eval again with temperature=0.3
```

Resume support is keyed by:

```
(run_id, sentence_id, variant_id, representation, dimension, model, temperature)
```

Running the same evaluation twice will skip any already-written rows.

## Analysis

### Load and compute deltas

```python
from typography_eval.src import analysis

results = analysis.load_results(run_id="run_000")
delta = analysis.compute_delta(results)
```

### Summaries

```python
by_dim = analysis.summarize_by_dimension(delta)
by_cat = analysis.summarize_by_sentence_category(delta)
by_var = analysis.summarize_by_variant(delta)
```

### Compare runs (e.g., temp 0.0 vs 0.3)

```python
cmp = analysis.compare_runs("run_000", "run_001")
print(cmp["dimension"]["pearson"], cmp["dimension"]["spearman"])
```

## Notes on backward compatibility

- Image paths remain `images/sentence_{sid:03d}/{variant_id}.png`.
- `metadata.csv` can be converted to `data/artifacts_v1.csv` via `io_utils.artifacts_from_metadata()`.
- OCR text files are expected under `ocr/` as `sentence_{sid:03d}.txt`.

Update `config.py` if your asset locations differ.
