# Analysis Summary

## Run Metadata

- **Run ID**: run_20260117_164819
- **Total responses**: 3960
- **Paired comparisons (dimensions)**: 3240
- **Paired comparisons (decision)**: 324
- **Inference mode**: both
- **Temperature**: 0.0

## Headline Results - Dimensions Mode

- **Overall flip rate**: 0.162 (95% CI: [0.145, 0.178], n=3240.0)

- **Overall approval rate (image)**: 0.394 (95% CI: [0.365, 0.421], n=3240.0)

- **Approval rate (OCR baseline)**: 0.383
- **Approval rate (Image variants)**: 0.394

### Flip Directionality

- **NO→YES**: 279 flips (53.2%)
- **YES→NO**: 245 flips (46.8%)

## Top Variants by Flip Rate

- **T7_comic**: 0.219
- **A1_opendyslexic_regular**: 0.189
- **T3_arial_regular**: 0.161
- **T5_arial_all_caps**: 0.156
- **T2_times_bold**: 0.156
- **T4_arial_bold**: 0.150
- **T1_times_regular**: 0.147
- **T6_monospace**: 0.144
- **T8_small_text**: 0.133

## Top Variants by Approval Rate (Image)

- **T5_arial_all_caps**: 0.444 (Text: 0.383)
- **T2_times_bold**: 0.428 (Text: 0.383)
- **T4_arial_bold**: 0.400 (Text: 0.383)
- **T3_arial_regular**: 0.400 (Text: 0.383)
- **T1_times_regular**: 0.397 (Text: 0.383)
- **T8_small_text**: 0.394 (Text: 0.383)
- **T6_monospace**: 0.394 (Text: 0.383)
- **A1_opendyslexic_regular**: 0.350 (Text: 0.383)
- **T7_comic**: 0.336 (Text: 0.383)

## Top Boundary Sentences

- Sentence 30: 0.322
- Sentence 7: 0.311
- Sentence 23: 0.267
- Sentence 29: 0.256
- Sentence 10: 0.233
- Sentence 12: 0.222
- Sentence 15: 0.222
- Sentence 16: 0.211
- Sentence 36: 0.200
- Sentence 9: 0.200

## Figures

### Flip Rates
- [Flip Rate by Variant](figures/flip_rate_by_variant.png)
- [Heatmap: Sentence x Variant](figures/heatmap_sentence_variant.png)

### Approval Rates
- [Approval Rate by Variant](figures/approval_rate_by_variant.png)
- [Approval Rate by Dimension](figures/approval_rate_by_dimension.png)

### Flip Directionality
- [Directionality Heatmap](figures/bias_direction_heatmap.png)

---

## Decision Mode Results

Analysis of escalation decisions (binary: escalate vs. don't escalate).

- **Decision flip rate**: 0.111 (95% CI: [0.042, 0.190], n=324.0)

- **Decision approval rate (image)**: 0.000 (95% CI: [0.000, 0.000], n=324.0)

- **Decision approval rate (OCR baseline)**: 0.111
- **Decision approval rate (Image variants)**: 0.000

### Decision Flip Directionality

- **YES→NO**: 36 flips (100.0%)

### Top Variants by Decision Flip Rate

- **A1_opendyslexic_regular**: 0.111
- **T1_times_regular**: 0.111
- **T2_times_bold**: 0.111
- **T3_arial_regular**: 0.111
- **T4_arial_bold**: 0.111
- **T5_arial_all_caps**: 0.111
- **T6_monospace**: 0.111
- **T7_comic**: 0.111
- **T8_small_text**: 0.111

### Top Variants by Decision Approval Rate

- **A1_opendyslexic_regular**: 0.000 (Text: 0.111)
- **T1_times_regular**: 0.000 (Text: 0.111)
- **T2_times_bold**: 0.000 (Text: 0.111)
- **T3_arial_regular**: 0.000 (Text: 0.111)
- **T4_arial_bold**: 0.000 (Text: 0.111)
- **T5_arial_all_caps**: 0.000 (Text: 0.111)
- **T6_monospace**: 0.000 (Text: 0.111)
- **T7_comic**: 0.000 (Text: 0.111)
- **T8_small_text**: 0.000 (Text: 0.111)

### Decision Mode Figures

- [Decision Flip Rate by Variant](figures/decision_flip_rate_by_variant.png)
- [Decision Approval Rate by Variant](figures/decision_approval_rate_by_variant.png)
- [Decision Directionality Heatmap](figures/decision_bias_direction_heatmap.png)
