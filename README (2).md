# Analysis-check utilities

These utilities write tabular outputs only. They document the composite barcode-complexity score and the sensitivity of the main summaries to analysis choices.

## `cssv_mosaic_complexity_score_audit.py`

Audits the three-component mosaic-complexity score:

```text
(1 - dominant-label purity) + normalized barcode entropy + switch rate
```

Raw label entropy is expressed in bits and, when needed, normalized by `log2(mosaic_k)`. ORF–mosaic mismatch remains available as a separate diagnostic in the merged input table but is not included in the score.

```bash
python analysis_checks/cssv_mosaic_complexity_score_audit.py \
  --merged "results/tree_mosaic_agreement/mosaic_orf_merged_per_genome.csv" \
  --out_dir "results/mosaic_complexity_score_audit" \
  --top_n 10 \
  --mosaic_k 8
```

Outputs:

- `mosaic_complexity_score_formula_components.csv`
- `mosaic_complexity_score_distribution_summary.csv`
- `mosaic_complexity_score_weight_sensitivity.csv`
- `mosaic_complexity_score_topN_by_weight_scheme.csv`

## `cssv_parameter_sensitivity.py`

Varies k-mer length, window length, step size, barcode-cluster K, and ORF-cluster K while keeping the two sources of sensitivity separate.

## `cssv_orf_threshold_sensitivity.py`

Repeats ORF prediction and ORF-boundary enrichment analyses across user-specified minimum ORF-length thresholds.
