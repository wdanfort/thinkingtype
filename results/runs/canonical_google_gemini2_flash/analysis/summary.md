# Typography Evaluation Analysis Summary

**Run ID:** run_20260117_164819
**Provider:** google
**Model:** models/gemini-2.0-flash

## Dimension Analysis

- **Overall Flip Rate:** 16.2% (95% CI: [14.5%, 17.8%])
- **Total Comparisons:** 3240
- **Top Flipping Dimension:** trustworthy (48.1%)
- **Top Flipping Variant:** T7_comic (21.9%)

### Top Dimensions by Flip Rate

| Dimension | Flip Rate | 95% CI | Total |
|-----------|-----------|--------|-------|
| trustworthy | 48.1% | [36.4%, 60.6%] | 324 |
| confident | 28.1% | [19.0%, 37.0%] | 324 |
| formal | 19.4% | [14.5%, 23.7%] | 324 |
| professional | 17.9% | [8.7%, 26.5%] | 324 |
| persuasive | 17.0% | [8.8%, 24.3%] | 324 |

### Direction Bias (Top Dimensions)

| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |
|-----------|-------------|---------|---------|----------|
| trustworthy | 156 | 96% | 4% | +92% |
| confident | 91 | 23% | 77% | -54% |
| formal | 63 | 5% | 95% | -90% |
| professional | 58 | 0% | 100% | -100% |
| persuasive | 55 | 96% | 4% | +93% |

## Decision Analysis

- **Decision Flip Rate:** 11.1% (95% CI: [4.2%, 19.0%])
- **Total Decisions:** 324
- **Direction Net Bias:** -100.0%
- **Interpretation:** Vision less likely to escalate

### Key Dimension-Decision Correlations

| Dimension | Lift |
|-----------|------|
| trustworthy | 0.00x |
| confident | 2.86x |
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
