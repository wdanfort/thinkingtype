# Analysis Summary

## Run Metadata

- **Run ID**: run_20260115_222423
- **Total responses**: 3960
- **Paired comparisons (dimensions)**: 3238
- **Paired comparisons (decision)**: 324
- **Inference mode**: both
- **Temperature**: 0.0

## Headline Results - Dimensions Mode

- **Overall flip rate**: 0.219 (95% CI: [0.187, 0.248], n=3238.0)

- **Overall approval rate (image)**: 0.285 (95% CI: [0.256, 0.314], n=3238.0)

- **Approval rate (OCR baseline)**: 0.442
- **Approval rate (Image variants)**: 0.285

### Flip Directionality

- **NO→YES**: 100 flips (14.1%)
- **YES→NO**: 608 flips (85.9%)

## Top Variants by Flip Rate

- **A1_opendyslexic_regular**: 0.284
- **T7_comic**: 0.256
- **T5_arial_all_caps**: 0.225
- **T4_arial_bold**: 0.211
- **T6_monospace**: 0.203
- **T1_times_regular**: 0.201
- **T2_times_bold**: 0.197
- **T3_arial_regular**: 0.197
- **T8_small_text**: 0.194

## Top Variants by Approval Rate (Image)

- **T8_small_text**: 0.314 (Text: 0.442)
- **T1_times_regular**: 0.306 (Text: 0.440)
- **T5_arial_all_caps**: 0.300 (Text: 0.442)
- **T2_times_bold**: 0.300 (Text: 0.442)
- **T4_arial_bold**: 0.297 (Text: 0.442)
- **T3_arial_regular**: 0.294 (Text: 0.442)
- **T6_monospace**: 0.283 (Text: 0.442)
- **T7_comic**: 0.247 (Text: 0.442)
- **A1_opendyslexic_regular**: 0.220 (Text: 0.443)

## Top Boundary Sentences

- Sentence 8: 0.611
- Sentence 20: 0.422
- Sentence 27: 0.400
- Sentence 7: 0.367
- Sentence 22: 0.367
- Sentence 32: 0.322
- Sentence 15: 0.311
- Sentence 13: 0.311
- Sentence 6: 0.289
- Sentence 10: 0.270

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

- **Decision flip rate**: 0.210 (95% CI: [0.105, 0.312], n=324.0)

- **Decision approval rate (image)**: 0.040 (95% CI: [0.000, 0.069], n=324.0)

- **Decision approval rate (OCR baseline)**: 0.194
- **Decision approval rate (Image variants)**: 0.040

### Decision Flip Directionality

- **NO→YES**: 9 flips (13.2%)
- **YES→NO**: 59 flips (86.8%)

### Top Variants by Decision Flip Rate

- **T2_times_bold**: 0.222
- **T8_small_text**: 0.222
- **T3_arial_regular**: 0.222
- **T4_arial_bold**: 0.222
- **T6_monospace**: 0.222
- **T7_comic**: 0.222
- **A1_opendyslexic_regular**: 0.194
- **T1_times_regular**: 0.194
- **T5_arial_all_caps**: 0.167

### Top Variants by Decision Approval Rate

- **T5_arial_all_caps**: 0.083 (Text: 0.194)
- **A1_opendyslexic_regular**: 0.056 (Text: 0.194)
- **T1_times_regular**: 0.056 (Text: 0.194)
- **T3_arial_regular**: 0.028 (Text: 0.194)
- **T2_times_bold**: 0.028 (Text: 0.194)
- **T4_arial_bold**: 0.028 (Text: 0.194)
- **T6_monospace**: 0.028 (Text: 0.194)
- **T7_comic**: 0.028 (Text: 0.194)
- **T8_small_text**: 0.028 (Text: 0.194)

### Decision Mode Figures

- [Decision Flip Rate by Variant](figures/decision_flip_rate_by_variant.png)
- [Decision Approval Rate by Variant](figures/decision_approval_rate_by_variant.png)
- [Decision Directionality Heatmap](figures/decision_bias_direction_heatmap.png)
