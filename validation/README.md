# Validation modules

These command-line modules provide tabular checks for the CSSV genome-geometry and sliding-window barcode workflow. They do not create manuscript figures.

## Full driver

`run_cssv_analysis_and_validation.py` runs the corrected circular barcode baseline and the validation modules in a clean output directory.

```bash
python validation/run_cssv_analysis_and_validation.py \
  --repo "." \
  --input_dir "data/core48_fasta" \
  --genbank_dir "data/core48_genbank" \
  --out_dir "results/full_analysis" \
  --existing_alignment "path/to/alignment.trim_gap0.50.faa" \
  --bootstrap_replicates 1000 \
  --force
```

The driver expects exactly 48 accession-specific sequence files. Combined FASTA files must be stored outside `--input_dir`.

## Modules and outputs

### Published-species validation

`cssv_external_species_validation.py`

Compares average-linkage clusters derived from the whole-genome k-mer cosine-distance matrix with the published species labels in `resources/core48_published_species.tsv`.

Outputs:
- `published_species_assignments_core48.csv`
- `external_species_cluster_mapping.csv`
- `external_species_contingency.csv`
- `external_species_validation_summary.csv`

### MDS diagnostics

`cssv_mds_diagnostics.py`

Calculates raw stress, Kruskal Stress-1, and correlations between original pairwise distances and distances in the two-dimensional MDS embedding.

Outputs:
- `mds_embedding_diagnostics.csv`
- `mds_distance_reconstruction.csv`

### Barcode clustering diagnostics

`cssv_barcode_clustering_diagnostics.py`

Evaluates K with silhouette, Calinski–Harabasz, Davies–Bouldin, objective, and cross-seed adjusted Rand index statistics. It also compares the baseline Euclidean KMeans partition with spherical/cosine k-means on L2-normalized k-mer-frequency vectors.

Outputs include:
- `barcode_k_seed_metrics.csv`
- `barcode_k_selection_and_stability_summary.csv`
- `barcode_best_k_by_criterion.csv`
- baseline Euclidean-versus-spherical assignment and per-genome metric comparisons;
- `baseline_assignment_reproduction_check.csv` confirming reproduction of the core K=8 assignments

### ORF3 NJ bootstrap

`cssv_orf3_nj_bootstrap.py`

Resamples amino-acid alignment columns, reconstructs pairwise-identity NJ trees, and reports support for non-trivial unrooted splits. No biological outgroup is assumed. A midpoint-rooted copy is written for display only.

Outputs:
- `orf3_nj_bootstrap_unrooted.newick`
- `orf3_nj_bootstrap_midpoint_display.newick`
- `orf3_nj_bootstrap_support.csv`
- `orf3_nj_bootstrap_summary.csv`

### ORF-boundary threshold sensitivity

`cssv_orf_boundary_threshold_sensitivity.py`

Repeats the pre-specified one-sided enrichment test at user-defined distance thresholds. The default driver tests 100, 200, and 400 bp.

Output:
- `orf_boundary_threshold_sensitivity_summary.csv`

### Accession, ORF3, and record-based EVE screen

`cssv_orf3_eve_record_audit.py`

Audits accession-level GenBank descriptions and source qualifiers for endogenous/integrated/host-flanking indicators and compares the predicted longest ORF with the annotated ORF3/polyprotein CDS. This is a record-based screen and does not experimentally establish episomal status.

Outputs:
- `accession_record_and_eve_screen.csv`
- `orf3_annotation_audit.csv`
- `accession_inclusion_and_orf3_audit.csv`
- `accession_orf3_eve_audit_summary.csv`

## Ranking output and ORF–mosaic diagnostics

The full driver writes a three-component `mosaic_complexity_score` based on low dominant-label purity, barcode entropy normalized by `log2(K)`, and circular switch rate. ORF–mosaic agreement, cluster mapping, and mismatch status are retained as separate diagnostic columns and tables rather than being added to the ranking score.
