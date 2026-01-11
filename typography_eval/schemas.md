# Typography eval schemas

## data/sentences_v1.csv

- `sentence_id` (int)
- `text` (str)
- `sentence_category` (str)
- `sentence_set_id` (str)

## data/variants_v1.csv

- `variant_id` (str)
- `font_family` (str)
- `font_path` (str)
- `size` (int)
- `uppercase` (bool)
- `variant_group` (str)
- `variant_set_id` (str)

## data/artifacts_v1.csv

- `artifact_id` (str)
- `sentence_id` (int)
- `variant_id` (str)
- `representation` (str)
- `image_path` (str)
- `sentence_set_id` (str)
- `variant_set_id` (str)

## runs/results.csv

- `run_id` (str)
- `sentence_id` (int)
- `variant_id` (str)
- `representation` (str)
- `dimension` (str)
- `model` (str)
- `temperature` (float)
- `response_raw` (str)
- `response_norm` (str)
- `response_01` (int)
- `created_at` (str)

## runs/runs.csv

- `run_id` (str)
- `created_at` (str)
- `model_text` (str)
- `model_image` (str)
- `temperature` (float)
- `sentence_set_id` (str)
- `variant_set_id` (str)
- `notes` (str)
