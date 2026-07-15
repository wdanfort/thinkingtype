# Typography Evaluation Analysis Summary

**Run ID:** v0c_release
**Provider:** openai
**Model:** gpt-5.6-sol

## Dimension Analysis

- **Overall Flip Rate:** 10.7% (95% CI: [9.4%, 12.1%])
- **Total Comparisons:** 9600
- **Top Flipping Dimension:** confident (31.7%)
- **Top Flipping Variant:** T7_comic (14.2%)

### Top Dimensions by Flip Rate

| Dimension | Flip Rate | 95% CI | Total |
|-----------|-----------|--------|-------|
| confident | 31.7% | [24.9%, 38.2%] | 960 |
| trustworthy | 20.1% | [14.8%, 26.1%] | 960 |
| formal | 16.6% | [13.2%, 20.4%] | 960 |
| persuasive | 11.4% | [7.0%, 15.9%] | 960 |
| high_risk | 11.0% | [6.5%, 16.1%] | 960 |

### Direction Bias (Top Dimensions)

| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |
|-----------|-------------|---------|---------|----------|
| confident | 304 | 97% | 3% | +95% |
| trustworthy | 193 | 41% | 59% | -17% |
| formal | 159 | 16% | 84% | -67% |
| persuasive | 109 | 73% | 27% | +47% |
| high_risk | 106 | 33% | 67% | -34% |

## Decision Analysis

- **Decision Flip Rate:** 1.2% (95% CI: [0.3%, 2.4%])
- **Total Decisions:** 960
- **Direction Net Bias:** +33.3%
- **Interpretation:** Mixed

### Key Dimension-Decision Correlations

| Dimension | Lift |
|-----------|------|
| emotional | 87.18x |
| urgent | 10.93x |
| immediate_action | 9.30x |

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
