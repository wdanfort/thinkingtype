# Typography Evaluation Analysis Summary

**Run ID:** canonical_google_gemini2_flash
**Provider:** google
**Model:** models/gemini-2.0-flash

## Dimension Analysis

- **Overall Flip Rate:** 16.5% (95% CI: [14.8%, 18.2%])
- **Total Comparisons:** 2880
- **Top Flipping Dimension:** trustworthy (48.6%)
- **Top Flipping Variant:** T7_comic (21.9%)

### Top Dimensions by Flip Rate

| Dimension | Flip Rate | 95% CI | Total |
|-----------|-----------|--------|-------|
| trustworthy | 48.6% | [36.9%, 61.0%] | 288 |
| confident | 28.8% | [19.6%, 38.0%] | 288 |
| formal | 21.2% | [15.9%, 25.6%] | 288 |
| professional | 17.7% | [8.7%, 26.2%] | 288 |
| persuasive | 17.4% | [9.1%, 25.0%] | 288 |

### Direction Bias (Top Dimensions)

| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |
|-----------|-------------|---------|---------|----------|
| trustworthy | 140 | 96% | 4% | +91% |
| confident | 83 | 25% | 75% | -49% |
| formal | 61 | 3% | 97% | -93% |
| professional | 51 | 0% | 100% | -100% |
| persuasive | 50 | 96% | 4% | +92% |

## Decision Analysis

- **Decision Flip Rate:** 11.1% (95% CI: [4.2%, 19.0%])
- **Total Decisions:** 288
- **Direction Net Bias:** -100.0%
- **Interpretation:** Vision less likely to escalate

### Key Dimension-Decision Correlations

| Dimension | Lift |
|-----------|------|
| trustworthy | 0.00x |
| confident | 3.18x |
| emotional | 0.00x |

## Output Files

**CSVs:**
- `flip_rates.csv` - Flip rates by dimension and variant with bootstrap CIs
- `bias_direction.csv` - Direction bias (NO→YES vs YES→NO)
- `decision_analysis.csv` - Decision-specific metrics and correlations
- `headline_metrics.json` - Key metrics for programmatic access

**Figures:**
- `fig1_flip_by_dimension.png` - Flip rates by dimension
- `fig2_flip_by_variant.png` - Flip rates by typography variant
- `fig3_direction_by_dimension.png` - Direction bias by dimension
- `fig4_direction_by_variant.png` - Direction bias by variant
- `fig5_decision_flip.png` - Decision flip rate and direction
