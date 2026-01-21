# Typography Evaluation Analysis Summary

**Run ID:** canonical_openai_gpt4o
**Provider:** openai
**Model:** gpt-4o

## Dimension Analysis

- **Overall Flip Rate:** 22.2% (95% CI: [19.0%, 25.1%])
- **Total Comparisons:** 2878
- **Top Flipping Dimension:** trustworthy (48.3%)
- **Top Flipping Variant:** A1_opendyslexic_regular (28.4%)

### Top Dimensions by Flip Rate

| Dimension | Flip Rate | 95% CI | Total |
|-----------|-----------|--------|-------|
| trustworthy | 48.3% | [37.0%, 58.9%] | 288 |
| high_risk | 43.9% | [32.5%, 55.4%] | 287 |
| confident | 35.8% | [25.6%, 46.0%] | 288 |
| professional | 30.3% | [18.5%, 40.8%] | 287 |
| formal | 29.2% | [22.9%, 34.8%] | 288 |

### Direction Bias (Top Dimensions)

| Dimension | Total Flips | NO→YES | YES→NO | Net Bias |
|-----------|-------------|---------|---------|----------|
| trustworthy | 139 | 0% | 100% | -100% |
| high_risk | 126 | 0% | 100% | -100% |
| confident | 103 | 23% | 77% | -53% |
| professional | 87 | 0% | 100% | -100% |
| formal | 84 | 0% | 100% | -100% |

## Decision Analysis

- **Decision Flip Rate:** 20.8% (95% CI: [10.4%, 31.0%])
- **Total Decisions:** 288
- **Direction Net Bias:** -73.3%
- **Interpretation:** Vision less likely to escalate

### Key Dimension-Decision Correlations

| Dimension | Lift |
|-----------|------|
| high_risk | 5.11x |
| professional | 3.87x |
| urgent | 5.38x |

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
