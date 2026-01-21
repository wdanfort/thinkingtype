# Typography Evaluation Analysis Summary

**Run ID:** canonical_anthropic_claude_sonnet4
**Provider:** anthropic
**Model:** claude-sonnet-4-20250514

## Dimension Analysis

- **Overall Flip Rate:** 15.7% (95% CI: [13.4%, 18.1%])
- **Total Comparisons:** 2880
- **Top Flipping Dimension:** trustworthy (55.6%)
- **Top Flipping Variant:** T5_arial_all_caps (23.6%)

### Top Dimensions by Flip Rate

| Dimension | Flip Rate | 95% CI | Total |
|-----------|-----------|--------|-------|
| trustworthy | 55.6% | [45.0%, 65.8%] | 288 |
| high_risk | 44.1% | [33.1%, 55.6%] | 288 |
| persuasive | 12.8% | [5.6%, 18.8%] | 288 |
| professional | 12.8% | [5.4%, 19.0%] | 288 |
| urgent | 12.5% | [6.5%, 17.6%] | 288 |

### Direction Bias (Top Dimensions)

| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |
|-----------|-------------|---------|---------|----------|
| trustworthy | 160 | 0% | 100% | -100% |
| high_risk | 127 | 1% | 99% | -98% |
| persuasive | 37 | 59% | 41% | +19% |
| professional | 37 | 73% | 27% | +46% |
| urgent | 36 | 100% | 0% | +100% |

## Decision Analysis

- **Decision Flip Rate:** 4.9% (95% CI: [0.0%, 8.3%])
- **Total Decisions:** 288
- **Direction Net Bias:** +0.0%
- **Interpretation:** None

### Key Dimension-Decision Correlations

| Dimension | Lift |
|-----------|------|
| emotional | 22.30x |
| persuasive | 1.85x |
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
