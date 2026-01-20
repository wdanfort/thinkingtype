# Typography Evaluation Analysis Summary

**Run ID:** run_20260115_222423
**Provider:** openai
**Model:** gpt-4o

## Dimension Analysis

- **Overall Flip Rate:** 21.9% (95% CI: [18.7%, 24.8%])
- **Total Comparisons:** 3238
- **Top Flipping Dimension:** trustworthy (47.2%)
- **Top Flipping Variant:** A1_opendyslexic_regular (28.4%)

### Top Dimensions by Flip Rate

| Dimension | Flip Rate | 95% CI | Total |
|-----------|-----------|--------|-------|
| trustworthy | 47.2% | [36.0%, 57.8%] | 324 |
| high_risk | 43.3% | [32.1%, 54.6%] | 323 |
| confident | 35.2% | [25.0%, 45.0%] | 324 |
| professional | 30.3% | [18.5%, 40.9%] | 323 |
| formal | 27.2% | [20.9%, 32.8%] | 324 |

### Direction Bias (Top Dimensions)

| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |
|-----------|-------------|---------|---------|----------|
| trustworthy | 153 | 0% | 100% | -100% |
| high_risk | 140 | 0% | 100% | -100% |
| confident | 114 | 23% | 77% | -54% |
| professional | 98 | 0% | 100% | -100% |
| formal | 88 | 0% | 100% | -100% |

## Decision Analysis

- **Decision Flip Rate:** 21.0% (95% CI: [10.5%, 31.2%])
- **Total Decisions:** 324
- **Direction Net Bias:** -73.5%
- **Interpretation:** Vision less likely to escalate

### Key Dimension-Decision Correlations

| Dimension | Lift |
|-----------|------|
| high_risk | 4.62x |
| professional | 3.86x |
| urgent | 5.34x |

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
