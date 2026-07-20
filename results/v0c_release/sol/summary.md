# Typography Evaluation Analysis Summary

**Run ID:** v0c_release_sol
**Provider:** openai
**Model:** gpt-5.6-sol

## Dimension Analysis

- **Overall Flip Rate:** 13.6% (95% CI: [12.2%, 14.9%])
- **Total Comparisons:** 9600
- **Top Flipping Dimension:** confident (51.6%)
- **Top Flipping Variant:** T7_comic (17.0%)

### Top Dimensions by Flip Rate

| Dimension | Flip Rate | 95% CI | Total |
|-----------|-----------|--------|-------|
| confident | 51.6% | [43.7%, 58.8%] | 960 |
| trustworthy | 18.2% | [12.8%, 24.2%] | 960 |
| formal | 17.6% | [14.4%, 21.1%] | 960 |
| persuasive | 16.6% | [11.1%, 22.1%] | 960 |
| high_risk | 12.2% | [7.7%, 17.3%] | 960 |

### Direction Bias (Top Dimensions)

| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |
|-----------|-------------|---------|---------|----------|
| confident | 495 | 98% | 2% | +97% |
| trustworthy | 175 | 39% | 61% | -21% |
| formal | 169 | 25% | 75% | -49% |
| persuasive | 159 | 74% | 26% | +47% |
| high_risk | 117 | 39% | 61% | -21% |

## Decision Analysis

- **Decision Flip Rate:** 2.7% (95% CI: [0.9%, 5.0%])
- **Total Decisions:** 960
- **Direction Net Bias:** +38.5%
- **Interpretation:** Mixed

### Key Dimension-Decision Correlations

| Dimension | Lift |
|-----------|------|
| emotional | 0.00x |
| formal | 0.85x |
| trustworthy | 0.82x |

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
