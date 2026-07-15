# Typography Evaluation Analysis Summary

**Run ID:** v0c_release_fable
**Provider:** anthropic
**Model:** claude-fable-5

## Dimension Analysis

- **Overall Flip Rate:** 8.6% (95% CI: [7.4%, 10.0%])
- **Total Comparisons:** 9367
- **Top Flipping Dimension:** confident (19.2%)
- **Top Flipping Variant:** T7_comic (10.8%)

### Top Dimensions by Flip Rate

| Dimension | Flip Rate | 95% CI | Total |
|-----------|-----------|--------|-------|
| confident | 19.2% | [14.0%, 24.7%] | 938 |
| trustworthy | 16.4% | [11.1%, 22.2%] | 922 |
| high_risk | 10.8% | [6.6%, 15.5%] | 927 |
| persuasive | 10.4% | [6.5%, 14.6%] | 956 |
| formal | 9.7% | [6.3%, 13.6%] | 938 |

### Direction Bias (Top Dimensions)

| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |
|-----------|-------------|---------|---------|----------|
| confident | 180 | 82% | 18% | +64% |
| trustworthy | 151 | 58% | 42% | +17% |
| high_risk | 100 | 25% | 75% | -50% |
| persuasive | 99 | 85% | 15% | +70% |
| formal | 91 | 0% | 100% | -100% |

## Decision Analysis

- **Decision Flip Rate:** 2.7% (95% CI: [0.8%, 5.1%])
- **Total Decisions:** 844
- **Direction Net Bias:** -56.5%
- **Interpretation:** Vision less likely to escalate

### Key Dimension-Decision Correlations

| Dimension | Lift |
|-----------|------|
| urgent | 13.33x |
| high_risk | 7.95x |
| immediate_action | 3.38x |

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
