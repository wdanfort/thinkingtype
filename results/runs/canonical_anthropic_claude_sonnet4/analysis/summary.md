# Typography Evaluation Analysis Summary

**Run ID:** run_20260116_144444
**Provider:** anthropic
**Model:** claude-sonnet-4-20250514

## Dimension Analysis

- **Overall Flip Rate:** 15.6% (95% CI: [13.4%, 18.0%])
- **Total Comparisons:** 2910
- **Top Flipping Dimension:** trustworthy (55.3%)
- **Top Flipping Variant:** T5_arial_all_caps (23.6%)

### Top Dimensions by Flip Rate

| Dimension | Flip Rate | 95% CI | Total |
|-----------|-----------|--------|-------|
| trustworthy | 55.3% | [44.7%, 65.6%] | 291 |
| high_risk | 43.6% | [32.7%, 55.1%] | 291 |
| persuasive | 13.1% | [5.9%, 19.0%] | 291 |
| professional | 13.1% | [5.4%, 19.4%] | 291 |
| urgent | 12.4% | [6.5%, 17.4%] | 291 |

### Direction Bias (Top Dimensions)

| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |
|-----------|-------------|---------|---------|----------|
| trustworthy | 161 | 0% | 100% | -100% |
| high_risk | 127 | 1% | 99% | -98% |
| persuasive | 38 | 58% | 42% | +16% |
| professional | 38 | 71% | 29% | +42% |
| urgent | 36 | 100% | 0% | +100% |

## Decision Analysis

- **Decision Flip Rate:** 4.8% (95% CI: [0.0%, 8.3%])
- **Total Decisions:** 291
- **Direction Net Bias:** +0.0%
- **Interpretation:** None

### Key Dimension-Decision Correlations

| Dimension | Lift |
|-----------|------|
| emotional | 22.54x |
| persuasive | 1.82x |
| form_dependent | 0.00x |

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
