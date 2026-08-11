# cssv-geom-mosaic

Alignment-free genome geometry and sliding-window mosaic barcodes for characterizing modular diversity in the cacao swollen shoot virus (CSSV) complex, cross-validated against ORF3 phylogeny.

This repository provides a reproducible, command-line pipeline to:
1) compute alignment-free genome distances (k-mer cosine),
2) build sliding-window mosaic barcodes,
3) predict ORFs from sequence only (no GenBank CDS required),
4) derive ORF3 (longest ORF proxy) protein phylogeny distances and NJ tree,
5) quantify concordance between genome-scale distances and ORF3 distances,
6) quantify ORF–mosaic agreement and rank genomes by barcode-derived mosaic complexity,
7) **(Add-on A)** attach nearest-neighbor “context/provenance” labels (within-dataset; post-hoc),
8) **(Add-on B)** project a query set onto an expanded reference panel (e.g., a broader published cacao-infecting badnavirus panel; post-hoc).

Directory guide:
- `pipeline/` contains the core end-to-end analysis modules used to generate the main results.
- `scripts/` contains optional plotting and table-generation helpers.
- `analysis_checks/` contains CSV-only score-audit and parameter-sensitivity utilities.
- `validation/` contains external-label, clustering, MDS, ORF-boundary, ORF3-bootstrap, and accession-record validation modules.
- `tools/` contains accession-exact input-preparation utilities.
- `resources/` contains the published species labels used for external validation.
- `VERSION` and `CHANGELOG.md` identify the validated public release and summarize method-level changes.

---

## Data (NCBI)

Raw genome sequences are **NOT** included in this repository.

Please download the 48 CSSV genomes from NCBI using the accession list in:

- `ACCESSIONS.txt`

You may download sequences as `.gb/.gbk` (preferred) or `.fasta`.  
Place downloaded files into a single input folder, e.g.:

- `data/raw/`

> The pipeline can parse multi-record GenBank/FASTA files, but **one genome per file** is recommended for clarity.

---

## Installation

### Option A: Conda (recommended)

```bash
conda create -n cssv_geom python=3.11 -y
conda activate cssv_geom
pip install -r requirements.txt
```

### Option B: pip

```bash
pip install -r requirements.txt
```

---

## Quickstart (end-to-end)

Assume:
- raw genomes in: `data/raw/`
- outputs in: `results/`

### 1) Genome k-mer distances + MDS embedding + window barcodes

```bash
python pipeline/cssv_gb_pipeline.py   --input_dir "data/raw"   --out_dir "results/gb"   --k 4   --default_topology circular   --do_windows   --window 250   --step 50   --n_clusters 8
```

Key outputs:
- `results/gb/genome_summary.csv`
- `results/gb/k4_cosine_distance.csv`
- `results/gb/genome_embedding.csv`
- `results/gb/window_assignments.csv`

### 2) Mosaic switchpoints + ORF prediction + ORF-boundary null test

```bash
python pipeline/cssv_mosaic_orf_analysis.py   --input_dir "data/raw"   --window_assignments "results/gb/window_assignments.csv"   --out_dir "results/mosaic_orf"   --circular   --start_codons ATG   --min_orf_aa 300   --switchpoint_mode start   --min_run 2   --near_bp 200   --perm 2000   --plot_top 0   --no_plots
```

Key outputs:
- `results/mosaic_orf/predicted_orfs.csv`
- `results/mosaic_orf/switchpoints.csv`
- `results/mosaic_orf/switchpoint_orf_distances.csv`
- `results/mosaic_orf/enrichment_summary.csv`
- `results/mosaic_orf/null_fracs.npy`

### 3) Extract predicted ORF nucleotide/protein sequences

```bash
python pipeline/cssv_extract_predicted_orf_seqs.py   --input_dir "data/raw"   --predicted_orfs "results/mosaic_orf/predicted_orfs.csv"   --out_dir "results/mosaic_orf/orf_seqs"
```

Outputs:
- `results/mosaic_orf/orf_seqs/predicted_orfs.fna`
- `results/mosaic_orf/orf_seqs/predicted_orfs.faa`
- `results/mosaic_orf/orf_seqs/predicted_orfs_extracted.csv`

### 4) Select the longest ORF per genome (ORF3 proxy)

```bash
python pipeline/cssv_extract_longest_orfs.py   --predicted_orfs_csv "results/mosaic_orf/predicted_orfs.csv"   --orf_faa "results/mosaic_orf/orf_seqs/predicted_orfs.faa"   --orf_fna "results/mosaic_orf/orf_seqs/predicted_orfs.fna"   --out_dir "results/orf3"   --top_n 1
```

Outputs:
- `results/orf3/longest_orfs.faa`
- `results/orf3/longest_orfs.fna`
- `results/orf3/longest_orfs.csv`

---

## MAFFT step (run OUTSIDE the Python pipeline)

We intentionally keep MAFFT outside the main Python pipeline because installation and permissions can differ across platforms (Windows/WSL/Linux/macOS).

### A) Run MAFFT (recommended via WSL/Linux/macOS)

Align the ORF3 proteins:

```bash
mafft --auto --anysymbol "results/orf3/longest_orfs.faa" > "results/orf3/alignment.mafft.faa"
```

Sanity check:
- `results/orf3/alignment.mafft.faa` must be non-empty and start with `>`.

If you run MAFFT in WSL and see permission issues:

```bash
sudo mafft --auto --anysymbol "results/orf3/longest_orfs.faa"   1> "results/orf3/alignment.mafft.faa"   2> "results/orf3/mafft.stderr.log"
```

### B) Continue the pipeline using the precomputed alignment

```bash
python pipeline/cssv_orf_msa_tree.py   --faa "results/orf3/alignment.mafft.faa"   --out_dir "results/orf3_phylogeny"   --aligner none   --gap_cutoff 0.5
```

Outputs:
- `results/orf3_phylogeny/alignment.trim_gap0.50.faa`
- `results/orf3_phylogeny/pairwise_identity_distance.csv`
- `results/orf3_phylogeny/nj_tree.newick`
- `results/orf3_phylogeny/nj_tree.pdf`
- `results/orf3_phylogeny/pairwise_distance_heatmap.pdf`

---

## Concordance tests, ORF–mosaic agreement, and mosaic-complexity ranking

### 5) Distance–distance concordance (Mantel-style permutation)

```bash
python pipeline/cssv_compare_distances.py   --matrix_a "results/gb/k4_cosine_distance.csv"   --matrix_b "results/orf3_phylogeny/pairwise_identity_distance.csv"   --b_split "|"   --out_dir "results/compare_distances"   --method spearman   --perm 5000   --seed 0   --plot
```

### 6) ORF–mosaic agreement + mosaic-complexity ranking

```bash
python pipeline/cssv_tree_mosaic_agreement.py   --orf_dist "results/orf3_phylogeny/pairwise_identity_distance.csv"   --orf_name_split "|"   --window_assignments "results/gb/window_assignments.csv"   --out_dir "results/tree_mosaic_agreement"   --k_orf 8   --mosaic_k 8   --min_purity 0.6   --top_n 20   --tree_newick "results/orf3_phylogeny/nj_tree.newick"
```

Key outputs:
- `results/tree_mosaic_agreement/mosaic_ranked_genomes.csv`
- `results/tree_mosaic_agreement/mosaic_orf_merged_per_genome.csv`
- `results/tree_mosaic_agreement/agreement_metrics.csv`
- `results/tree_mosaic_agreement/contingency_orf_vs_mosaic.csv`

---

## Add-on A: nearest-neighbor context / provenance annotation (within-dataset; post-hoc)

This add-on step provides interpretable nearest-neighbor context labels **without altering any core analysis**.
It uses:
- the **core distance matrix** from the alignment-free genome geometry step, and
- **public record descriptors** (e.g., isolate names and regional descriptors) available in the corresponding NCBI record descriptions.

### Inputs

- Distance matrix (generated by core pipeline):
  - `results/gb/k4_cosine_distance.csv` *(or a compatible genome-by-genome distance matrix produced by this pipeline)*

- Header/description table:
  - created from a FASTA containing the accessions present in the analysis (see `addons/addon_01_reference_projection.py`)

### Outputs

- `nearest_neighbor_provenance.tsv`  
  Per-genome nearest neighbor mapping (query, nearest accession, distance) plus extracted context fields when available.

- `nearest_neighbor_token_summary.tsv`  
  Frequency summary of recurring isolate-like / regional descriptors among the nearest-neighbor matches.

These tables are intended for Supplementary reporting (e.g., Supplementary Data S1).

### Run (example)

```bash
python addons/addon_01_reference_projection.py --fasta combined_genomes.fasta

python addons/addon_02_provenance_join.py   --dist "results/gb/k4_cosine_distance.csv"   --headers "combined_genomes.header_richness.tsv"
```

Notes:
- `combined_genomes.fasta` is simply a combined FASTA file containing your genomes (e.g., concatenated from `data/raw/`).
- The add-on expects a distance matrix whose row/column IDs match the accessions in the header table.
- If your project uses a different distance matrix filename (e.g., `results/w2_distance.csv`), pass that path instead.

---

## Add-on B: reference-panel projection (query → nearest reference; post-hoc)

This add-on projects a **query set** onto a **reference panel** using an alignment-free k-mer cosine distance (k-mer frequency vectors).
It is useful when you want to contextualize a core dataset against a broader panel (e.g., PMPP-like reference collections) **without** re-running or re-tuning the core pipeline.

### Inputs

- Query FASTA: a multi-FASTA of genomes you want to label (e.g., `combined_genomes.fasta`)
- Reference FASTA: a multi-FASTA of reference genomes (e.g., `cibv.fa`)

> Important: If your query genomes are also present in the reference FASTA, the script will avoid self-hits when IDs match.

### Outputs

- `projection_nearest_reference.tsv`  
  Query → nearest reference accession, distance, and reference header-derived descriptor fields (when present).

- `projection_token_summary.tsv`  
  Frequency summary of recurring isolate-like / regional descriptors among nearest-reference matches.

These tables are intended for Supplementary reporting (e.g., Supplementary Data S1).

### Run (example)

```bash
python addons/addon_03_reference_panel_projection.py   --query_fasta combined_genomes.fasta   --ref_fasta cibv.fa   --k 4
```


---

## Analysis-check utilities (CSV only)

The `analysis_checks/` directory contains three tabular audit/sensitivity utilities used to document the mosaic-complexity score and parameter choices. These scripts do not generate figures. Their outputs are suitable for Supplementary Data S1.

### A) Mosaic-complexity score audit

Audits the three-component barcode-complexity score, the full score distribution, and simple component-weight perturbations. The input table is the merged output from the ORF–mosaic agreement step.

```bash
python analysis_checks/cssv_mosaic_complexity_score_audit.py \
  --merged "results/tree_mosaic_agreement/mosaic_orf_merged_per_genome.csv" \
  --out_dir "results/mosaic_complexity_score_audit" \
  --top_n 10 \
  --mosaic_k 8
```

Key outputs:
- `mosaic_complexity_score_formula_components.csv`
- `mosaic_complexity_score_distribution_summary.csv`
- `mosaic_complexity_score_weight_sensitivity.csv`
- `mosaic_complexity_score_topN_by_weight_scheme.csv`

### B) Parameter sensitivity

Runs one-at-a-time sensitivity tests around the baseline mosaic-barcode configuration and separately varies ORF-cluster K while holding the baseline mosaic barcode fixed.

```bash
python analysis_checks/cssv_parameter_sensitivity.py \
  --repo "." \
  --input_dir "data/raw" \
  --orf_dist "results/orf3_phylogeny/pairwise_identity_distance.csv" \
  --out_dir "results/parameter_sensitivity"
```

Key outputs:
- `sensitivity_summary.csv`
- `sensitivity_topN_by_config.csv`
- `sensitivity_topN_overlap_matrix.csv`

### C) ORF-threshold sensitivity

Repeats ORF prediction and ORF-boundary enrichment analyses across multiple minimum ORF-length thresholds.

```bash
python analysis_checks/cssv_orf_threshold_sensitivity.py \
  --repo "." \
  --input_dir "data/raw" \
  --window_assignments "results/gb/window_assignments.csv" \
  --out_dir "results/orf_threshold_sensitivity" \
  --thresholds "200,300,400"
```

Key output:
- `orf_threshold_sensitivity_summary.csv`

---

## Accession-exact input preparation

The input-preparation utility searches accession-specific local files first, then an optional multi-FASTA panel, and downloads only missing records from NCBI. It writes one record per file so combined FASTA files cannot be loaded accidentally as duplicate inputs.

```bash
python tools/cssv_prepare_core_inputs.py \
  --accessions ACCESSIONS.txt \
  --source_dir "path/to/local/sequence/files" \
  --panel_fasta "path/to/reference_panel.fasta" \
  --out_fasta_dir "data/core48_fasta" \
  --out_genbank_dir "data/core48_genbank" \
  --email "your.email@example.org" \
  --download_missing
```

The GenBank records are used for accession-level record screening and comparison of the predicted longest ORF with annotated ORF3/polyprotein features. This record-based screen does not experimentally establish episomal status.

---

## Full analysis and validation suite

The validation driver runs the corrected circular barcode workflow and writes tabular outputs only. Circular origin-spanning barcode windows and the last-to-first barcode boundary are included. Barcode entropy is reported in bits and normalized by `log2(K)` in the composite mosaic-complexity score. ORF–mosaic mismatch is retained as a separate diagnostic and is not included in the score.

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

If `--existing_alignment` is omitted, `cssv_orf_msa_tree.py` attempts native MAFFT and then WSL MAFFT. The validation outputs include:

- published-species agreement of whole-genome k-mer clusters;
- MDS Stress-1 and distance-preservation diagnostics;
- K-selection, seed-stability, and Euclidean-versus-spherical clustering checks;
- ORF3 neighbor-joining bootstrap support with no biological outgroup;
- ORF-boundary proximity sensitivity at 100, 200, and 400 bp;
- accession-level record/EVE flags and annotated-ORF3 consistency checks;
- score-audit, barcode-parameter, ORF-cluster-K, and ORF-length-threshold sensitivity tables.

See `validation/README.md` for output details.


---

## Paper-ready figures and Supplementary Data (Figure 1–7, Table 1, S1.xlsx)

After you have generated all core results (gb pipeline → mosaic/orf → ORF3 phylogeny → concordance → mosaic-complexity ranking),
you can build final paper-ready figures (PDF vector + PNG 300 dpi), Table 1 (Excel), and Supplementary Data S1 (single multi-sheet Excel).

Example (Windows; adjust paths as needed):

```bat
python paper\cssv_make_paper_package.py --gb_dir "gb_results" --mosaic_orf_dir "gb_results\mosaic_orf" --orf3_dir "gb_results\mosaic_orf\longest_orfs" --orf3_phylogeny_dir "gb_results\mosaic_orf\longest_orfs\phylogeny" --compare_dir "gb_results\distance_compare" --agreement_dir "gb_results\tree_mosaic_agreement" --switchpoint_post_dir "gb_results\switchpoint_post" --out_dir "paper_outputs" --top_n 10 --panel_case upper --dpi 300
```

---

## Reproducibility notes

- All steps are deterministic given fixed random seeds (where used).
- Raw inputs must match the accessions listed in `ACCESSIONS.txt`.
- Raw genomes are downloaded from NCBI using the accessions in `ACCESSIONS.txt`.
