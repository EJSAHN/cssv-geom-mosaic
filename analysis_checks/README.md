# Analysis-check utilities

These utilities generate CSV-only audit and sensitivity outputs for the CSSV genome-geometry/mosaic-barcode pipeline. They are intended for Supplementary Data S1 and do not generate figures.

## `cssv_mosaic_discordance_score_audit.py`
Audits the composite mosaic-discordance score formula, score distribution, and simple component-weight perturbations.

```bash
python analysis_checks/cssv_mosaic_discordance_score_audit.py \
  --merged "results/tree_mosaic_agreement/mosaic_orf_merged_per_genome.csv" \
  --out_dir "results/mosaic_discordance_score_audit" \
  --top_n 10
```

Outputs:
- `mosaic_discordance_score_formula_components.csv`
- `mosaic_discordance_score_distribution_summary.csv`
- `mosaic_discordance_score_weight_sensitivity.csv`
- `mosaic_discordance_score_topN_by_weight_scheme.csv`

## `cssv_parameter_sensitivity.py`
Runs parameter-sensitivity analyses. Mosaic-barcode parameters are varied one at a time while ORF-cluster K is held fixed, and ORF-cluster K is varied separately while the baseline mosaic barcode is held fixed.

```bash
python analysis_checks/cssv_parameter_sensitivity.py \
  --repo "." \
  --input_dir "data/raw" \
  --orf_dist "results/orf3_phylogeny/pairwise_identity_distance.csv" \
  --out_dir "results/parameter_sensitivity"
```

Outputs:
- `sensitivity_summary.csv`
- `sensitivity_topN_by_config.csv`
- `sensitivity_topN_overlap_matrix.csv`

## `cssv_orf_threshold_sensitivity.py`
Repeats ORF prediction and ORF-boundary enrichment analyses across several minimum ORF-length thresholds.

```bash
python analysis_checks/cssv_orf_threshold_sensitivity.py \
  --repo "." \
  --input_dir "data/raw" \
  --window_assignments "results/gb/window_assignments.csv" \
  --out_dir "results/orf_threshold_sensitivity" \
  --thresholds "200,300,400"
```

Output:
- `orf_threshold_sensitivity_summary.csv`
