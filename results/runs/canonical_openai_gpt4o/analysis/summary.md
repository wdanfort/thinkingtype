# Analysis Summary

## Run Metadata

- **Run ID**: run_20260115_222423
- **Total responses**: 3564
- **Paired comparisons (dimensions)**: 2878
- **Paired comparisons (decision)**: 288
- **Inference mode**: both
- **Temperature**: 0.0

## Headline Results - Dimensions Mode

- **Overall flip rate**: 0.218 (95% CI: [0.188, 0.247], n=2878.0)

- **Overall approval rate (image)**: 0.281 (95% CI: [0.252, 0.311], n=2878.0)

- **Approval rate (OCR baseline)**: 0.442
- **Approval rate (Image variants)**: 0.281

### Flip Directionality

- **NO→YES**: 83 flips (13.2%)
- **YES→NO**: 545 flips (86.8%)

## Top Variants by Flip Rate

- **A1_opendyslexic_regular**: 0.284
- **T7_comic**: 0.256
- **T5_arial_all_caps**: 0.225
- **T4_arial_bold**: 0.206
- **T6_monospace**: 0.197
- **T1_times_regular**: 0.195
- **T3_arial_regular**: 0.192
- **T2_times_bold**: 0.192

## Top Variants by Approval Rate (Image)

- **T1_times_regular**: 0.306 (OCR: 0.440)
- **T2_times_bold**: 0.300 (OCR: 0.442)
- **T5_arial_all_caps**: 0.300 (OCR: 0.442)
- **T4_arial_bold**: 0.297 (OCR: 0.442)
- **T3_arial_regular**: 0.294 (OCR: 0.442)
- **T6_monospace**: 0.283 (OCR: 0.442)
- **T7_comic**: 0.247 (OCR: 0.442)
- **A1_opendyslexic_regular**: 0.220 (OCR: 0.443)

## Top Boundary Sentences

- Sentence 8: 0.613
- Sentence 20: 0.438
- Sentence 22: 0.362
- Sentence 7: 0.362
- Sentence 13: 0.312
- Sentence 15: 0.312
- Sentence 32: 0.312
- Sentence 6: 0.312
- Sentence 27: 0.300
- Sentence 10: 0.278

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

- **Decision flip rate**: 0.208 (95% CI: [0.104, 0.310], n=288.0)

- **Decision approval rate (image)**: 0.042 (95% CI: [0.000, 0.071], n=288.0)

- **Decision approval rate (OCR baseline)**: 0.194
- **Decision approval rate (Image variants)**: 0.042

### Decision Flip Directionality

- **NO→YES**: 8 flips (13.3%)
- **YES→NO**: 52 flips (86.7%)

### Top Variants by Decision Flip Rate

- **T3_arial_regular**: 0.222
- **T2_times_bold**: 0.222
- **T6_monospace**: 0.222
- **T4_arial_bold**: 0.222
- **T7_comic**: 0.222
- **T1_times_regular**: 0.194
- **A1_opendyslexic_regular**: 0.194
- **T5_arial_all_caps**: 0.167

### Top Variants by Decision Approval Rate

- **T5_arial_all_caps**: 0.083 (OCR: 0.194)
- **A1_opendyslexic_regular**: 0.056 (OCR: 0.194)
- **T1_times_regular**: 0.056 (OCR: 0.194)
- **T2_times_bold**: 0.028 (OCR: 0.194)
- **T3_arial_regular**: 0.028 (OCR: 0.194)
- **T4_arial_bold**: 0.028 (OCR: 0.194)
- **T6_monospace**: 0.028 (OCR: 0.194)
- **T7_comic**: 0.028 (OCR: 0.194)

### Decision Mode Figures

- [Decision Flip Rate by Variant](figures/decision_flip_rate_by_variant.png)
- [Decision Approval Rate by Variant](figures/decision_approval_rate_by_variant.png)
- [Decision Directionality Heatmap](figures/decision_bias_direction_heatmap.png)
