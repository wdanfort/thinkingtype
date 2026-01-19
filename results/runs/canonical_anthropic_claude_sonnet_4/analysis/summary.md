# Analysis Summary

## Run Metadata

- **Run ID**: run_20260116_144444
- **Total responses**: 3597
- **Paired comparisons (dimensions)**: 2910
- **Paired comparisons (decision)**: 291
- **Inference mode**: both
- **Temperature**: 0.0

## Headline Results - Dimensions Mode

- **Overall flip rate**: 0.156 (95% CI: [0.134, 0.180], n=2910.0)

- **Overall approval rate (image)**: 0.285 (95% CI: [0.260, 0.308], n=2910.0)

- **Approval rate (OCR baseline)**: 0.368
- **Approval rate (Image variants)**: 0.285

### Flip Directionality

- **NO→YES**: 106 flips (23.3%)
- **YES→NO**: 349 flips (76.7%)

## Top Variants by Flip Rate

- **T5_arial_all_caps**: 0.236
- **T7_comic**: 0.158
- **T6_monospace**: 0.150
- **T3_arial_regular**: 0.147
- **T2_times_bold**: 0.147
- **A1_opendyslexic_regular**: 0.139
- **T1_times_regular**: 0.139
- **T4_arial_bold**: 0.139
- **T8_small_text**: 0.100

## Top Variants by Approval Rate (Image)

- **T5_arial_all_caps**: 0.372 (Text: 0.369)
- **T4_arial_bold**: 0.286 (Text: 0.369)
- **T2_times_bold**: 0.278 (Text: 0.369)
- **T6_monospace**: 0.275 (Text: 0.369)
- **T1_times_regular**: 0.275 (Text: 0.369)
- **T3_arial_regular**: 0.272 (Text: 0.369)
- **A1_opendyslexic_regular**: 0.269 (Text: 0.369)
- **T7_comic**: 0.261 (Text: 0.369)
- **T8_small_text**: 0.167 (Text: 0.267)

## Top Boundary Sentences

- Sentence 34: 0.400
- Sentence 7: 0.338
- Sentence 23: 0.287
- Sentence 32: 0.275
- Sentence 1: 0.256
- Sentence 36: 0.237
- Sentence 21: 0.212
- Sentence 35: 0.200
- Sentence 24: 0.200
- Sentence 22: 0.200

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

- **Decision flip rate**: 0.048 (95% CI: [0.000, 0.083], n=291.0)

- **Decision approval rate (image)**: 0.027 (95% CI: [0.000, 0.047], n=291.0)

- **Decision approval rate (OCR baseline)**: 0.027
- **Decision approval rate (Image variants)**: 0.027

### Decision Flip Directionality

- **NO→YES**: 7 flips (50.0%)
- **YES→NO**: 7 flips (50.0%)

### Top Variants by Decision Flip Rate

- **T2_times_bold**: 0.056
- **T7_comic**: 0.056
- **T3_arial_regular**: 0.056
- **T4_arial_bold**: 0.056
- **T5_arial_all_caps**: 0.056
- **T6_monospace**: 0.056
- **A1_opendyslexic_regular**: 0.028
- **T1_times_regular**: 0.028
- **T8_small_text**: 0.000

### Top Variants by Decision Approval Rate

- **A1_opendyslexic_regular**: 0.056 (Text: 0.028)
- **T2_times_bold**: 0.028 (Text: 0.028)
- **T3_arial_regular**: 0.028 (Text: 0.028)
- **T4_arial_bold**: 0.028 (Text: 0.028)
- **T5_arial_all_caps**: 0.028 (Text: 0.028)
- **T7_comic**: 0.028 (Text: 0.028)
- **T6_monospace**: 0.028 (Text: 0.028)
- **T1_times_regular**: 0.000 (Text: 0.028)
- **T8_small_text**: 0.000 (Text: 0.000)

### Decision Mode Figures

- [Decision Flip Rate by Variant](figures/decision_flip_rate_by_variant.png)
- [Decision Approval Rate by Variant](figures/decision_approval_rate_by_variant.png)
- [Decision Directionality Heatmap](figures/decision_bias_direction_heatmap.png)
