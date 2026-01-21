# Analysis Pipeline Refactoring Plan

## New Function Structure

### Core Data Generation (Keep & Modify)
- `load_results()` - Keep as-is
- `build_delta_table()` - Keep as-is
- `coerce_to_binary()` - Keep (from prompts module)

### NEW: Consolidated CSV Generators
- `generate_flip_rates_csv(paired)` → flip_rates.csv
  - Consolidates: flip_overall, flip_by_dimension, flip_by_variant
  - Adds bootstrap CI columns directly

- `generate_bias_direction_csv(paired)` → bias_direction.csv
  - Consolidates: bias_direction_overall, by_dimension, by_variant
  - Removes: by_category, by_sentence

- `generate_decision_analysis_csv(paired, paired_dimensions)` → decision_analysis.csv
  - All decision-specific metrics in one file
  - Includes: flip_rate, direction, correlations, lift

- `generate_headline_metrics_json(paired, decision_paired, config, run_id)` → headline_metrics.json
  - Top-level summary stats for programmatic access

### NEW: Streamlined Figure Generators
- `plot_flip_by_dimension(flip_rates, path)` → fig1_flip_by_dimension.png
- `plot_flip_by_variant(flip_rates, path)` → fig2_flip_by_variant.png
- `plot_direction_by_dimension(bias_direction, path)` → fig3_direction_by_dimension.png
- `plot_direction_by_variant(bias_direction, path)` → fig4_direction_by_variant.png
- `plot_decision_flip(decision_analysis, path)` → fig5_decision_flip.png

### DELETE: Old Functions (to remove)
- `compute_flip_rates()` - replaced by generate_flip_rates_csv
- `compute_approval_rates()` - not needed in streamlined version
- `compute_flip_directionality()` - replaced by generate_bias_direction_csv
- `bootstrap_ci()` - integrated into generate_flip_rates_csv
- All old plot functions (replaced by 5 new ones)

### Modified: Main Pipeline
- `analyze_run()` - simplified to call new generators only
- `generate_summary_md()` - updated to reference new files

## File Output Mapping

### OLD (30+ files per run) → NEW (9 files per run)

**CSVs:**
- flip_overall.csv ----→ \
- flip_by_dimension.csv → } flip_rates.csv (consolidated)
- flip_by_variant.csv → /
- bootstrap_ci*.csv ----/

- bias_direction_overall.csv ----→ \
- bias_direction_by_dimension.csv → } bias_direction.csv (consolidated)
- bias_direction_by_variant.csv --/

- decision_flip_*.csv --------→ \
- decision_bias_direction_*.csv → } decision_analysis.csv (consolidated)
- decision_bootstrap_ci_*.csv --/

- (NEW) headline_metrics.json

**Removed (not needed):**
- All approval_rate_*.csv files
- All *_by_category.csv files
- All *_by_sentence.csv files
- All *_by_variant_bucket.csv files
- results_long_format.csv (vestigial from earlier iteration)

**Figures:**
- OLD: 8-10 figures → NEW: 5 figures
- Removed: heatmaps, approval_rate plots, bootstrap_ci plots
- New names: fig1, fig2, fig3, fig4, fig5

## Implementation Steps

1. ✅ Create backup of analysis.py
2. Create new consolidated CSV generator functions
3. Create new streamlined plot functions
4. Rewrite analyze_run() to use new functions
5. Update generate_summary_md() for new file references
6. Test on one run
7. Update comparison.py for new outputs
8. Regenerate all canonical runs
9. Update docs/index.md with new figure paths
10. Clean up old analysis files from git

## Testing Checklist

- [ ] flip_rates.csv has correct schema (group_type, group_id, flip_count, total, flip_rate, ci_low, ci_high)
- [ ] bias_direction.csv has direction metrics
- [ ] decision_analysis.csv has correlation and lift metrics
- [ ] headline_metrics.json is valid JSON with all required fields
- [ ] All 5 figures render correctly
- [ ] summary.md references correct files
- [ ] Comparison outputs work with new structure
