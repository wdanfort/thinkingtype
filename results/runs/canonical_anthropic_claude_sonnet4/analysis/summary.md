# Analysis Summary

## Run Metadata

- **Run ID**: run_20260116_144444
- **Total responses**: 3564
- **Paired comparisons (dimensions)**: 2880
- **Paired comparisons (decision)**: 288
- **Inference mode**: both
- **Temperature**: 0.0

## Headline Results - Dimensions Mode

- **Overall flip rate**: 0.159 (95% CI: [0.135, 0.183], n=2880.0)

- **Overall approval rate (image)**: 0.286 (95% CI: [0.261, 0.310], n=2880.0)

- **Approval rate (OCR baseline)**: 0.372
- **Approval rate (Image variants)**: 0.286

### Flip Directionality

- **NO→YES**: 105 flips (22.9%)
- **YES→NO**: 353 flips (77.1%)

## Top Variants by Flip Rate

- **T5_arial_all_caps**: 0.233
- **T7_comic**: 0.161
- **T6_monospace**: 0.153
- **T3_arial_regular**: 0.150
- **T2_times_bold**: 0.150
- **T1_times_regular**: 0.142
- **A1_opendyslexic_regular**: 0.142
- **T4_arial_bold**: 0.142

## Top Variants by Approval Rate (Image)

- **T5_arial_all_caps**: 0.372 (OCR: 0.372)
- **T4_arial_bold**: 0.286 (OCR: 0.372)
- **T2_times_bold**: 0.278 (OCR: 0.372)
- **T1_times_regular**: 0.275 (OCR: 0.372)
- **T6_monospace**: 0.275 (OCR: 0.372)
- **T3_arial_regular**: 0.272 (OCR: 0.372)
- **A1_opendyslexic_regular**: 0.269 (OCR: 0.372)
- **T7_comic**: 0.261 (OCR: 0.372)

## Top Boundary Sentences

- Sentence 34: 0.400
- Sentence 7: 0.338
- Sentence 23: 0.287
- Sentence 32: 0.275
- Sentence 22: 0.275
- Sentence 1: 0.263
- Sentence 36: 0.237
- Sentence 21: 0.212
- Sentence 24: 0.200
- Sentence 35: 0.200

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

- **Decision flip rate**: 0.049 (95% CI: [0.000, 0.083], n=288.0)

- **Decision approval rate (image)**: 0.028 (95% CI: [0.000, 0.048], n=288.0)

- **Decision approval rate (OCR baseline)**: 0.028
- **Decision approval rate (Image variants)**: 0.028

### Decision Flip Directionality

- **NO→YES**: 7 flips (50.0%)
- **YES→NO**: 7 flips (50.0%)

### Top Variants by Decision Flip Rate

- **T3_arial_regular**: 0.056
- **T2_times_bold**: 0.056
- **T5_arial_all_caps**: 0.056
- **T4_arial_bold**: 0.056
- **T6_monospace**: 0.056
- **T7_comic**: 0.056
- **A1_opendyslexic_regular**: 0.028
- **T1_times_regular**: 0.028

### Top Variants by Decision Approval Rate

- **A1_opendyslexic_regular**: 0.056 (OCR: 0.028)
- **T2_times_bold**: 0.028 (OCR: 0.028)
- **T3_arial_regular**: 0.028 (OCR: 0.028)
- **T4_arial_bold**: 0.028 (OCR: 0.028)
- **T6_monospace**: 0.028 (OCR: 0.028)
- **T5_arial_all_caps**: 0.028 (OCR: 0.028)
- **T7_comic**: 0.028 (OCR: 0.028)
- **T1_times_regular**: 0.000 (OCR: 0.028)

### Decision Mode Figures

- [Decision Flip Rate by Variant](figures/decision_flip_rate_by_variant.png)
- [Decision Approval Rate by Variant](figures/decision_approval_rate_by_variant.png)
- [Decision Directionality Heatmap](figures/decision_bias_direction_heatmap.png)
