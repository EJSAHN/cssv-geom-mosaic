# cssv-geom-mosaic

Alignment-free genome geometry and sliding-window mosaic barcodes for characterizing modular diversity in the cacao swollen shoot virus complex.

This repository provides a reproducible command-line workflow for 48 full-length cacao swollen shoot disease-associated badnavirus genomes. It combines whole-genome k-mer distances, circular sliding-window barcode summaries, sequence-only ORF prediction, an ORF3-based alignment axis, information-theoretic comparisons, and validation analyses supporting the associated study.

## Main capabilities

1. Compute whole-genome k-mer cosine distances and a metric multidimensional scaling (MDS) embedding.
2. Build circular sliding-window mosaic barcodes, including origin-spanning windows.
3. Quantify dominant-label purity, barcode entropy, and circular switch rate.
4. Predict ORFs directly from circular nucleotide sequences and select the longest ORF as the ORF3 proxy.
5. Build an ORF3 protein alignment, pairwise identity-distance matrix, and neighbor-joining tree.
6. Compare whole-genome and ORF3 distance structures by permutation-based rank correlation.
7. Calculate a three-component mosaic-complexity score:

```text
(1 - dominant-label purity) + normalized barcode entropy + circular switch rate
```

Barcode entropy is calculated in bits and normalized by `log2(K)`, where `K` is the total number of barcode clusters. ORF–mosaic agreement and mismatch are retained as separate diagnostics and are not included in the ranking score.

8. Run published-species validation, MDS diagnostics, barcode-clustering diagnostics, ORF3 bootstrap analysis, ORF-boundary threshold sensitivity, and accession-level ORF3/EVE record audits.
9. Project the 48-genome query set onto a broader reference panel for nearest-reference context.

## Repository structure

```text
ACCESSIONS.txt                 Core 48 accession list
pipeline/                      Main sequence-analysis steps
scripts/                       Optional plotting, post-processing, and table utilities
analysis_checks/               Score and parameter-sensitivity utilities
validation/                    External and diagnostic validation modules
resources/                     Published species labels for the core 48 accessions
tools/                         Exact-accession input preparation
addons/                        Optional nearest-reference annotation and projection
```

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`
- MAFFT v7 for de novo ORF3 protein alignment

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

MAFFT can be supplied natively or through Windows Subsystem for Linux. An existing aligned ORF3 protein FASTA can also be passed to the full driver.

## Data preparation

Raw genome sequences are not distributed in this repository. Download or stage the accessions listed in `ACCESSIONS.txt`.

The input-preparation utility writes one FASTA and one GenBank record per accession and can search local sequence folders, an optional multi-FASTA reference panel, and NCBI Entrez for missing records.

```bash
python tools/cssv_prepare_core_inputs.py \
  --accessions ACCESSIONS.txt \
  --source_dir "path/to/local/sequences" \
  --panel_fasta "path/to/reference_panel.fasta" \
  --out_fasta_dir "data/core48_fasta" \
  --out_genbank_dir "data/core48_genbank" \
  --email "your.email@example.org" \
  --download_missing
```

Expected staged inputs:

```text
data/core48_fasta/             48 accession-specific FASTA files
data/core48_genbank/           48 accession-specific GenBank files
data/combined_core48.fasta     Combined query FASTA, stored outside the input folder
data/combined_core48.gb        Combined GenBank file, stored outside the input folder
data/input_manifest.tsv        Input provenance and length summary
```

Combined files are deliberately stored outside the accession-specific input folders so the analysis pipeline cannot read the same accession more than once.

## Full analysis and validation

Run the complete workflow from the repository root:

```bash
python validation/run_cssv_analysis_and_validation.py \
  --repo "." \
  --input_dir "data/core48_fasta" \
  --genbank_dir "data/core48_genbank" \
  --out_dir "results/full_analysis" \
  --bootstrap_replicates 1000 \
  --force
```

To reuse an existing aligned ORF3 protein FASTA:

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

The full driver expects exactly 48 accession-specific sequence files in `--input_dir`. It performs the baseline analysis and then runs the validation modules.

### Baseline parameters

```text
whole-genome k-mer length       4
barcode window                  250 bp
barcode step                    50 bp
barcode clusters                K = 8
circular barcode windows        yes
circular last-to-first boundary yes
minimum ORF length              300 aa
ORF-boundary test distance      200 bp
ORF-boundary permutations       2,000
whole-genome/ORF3 permutations  5,000
ORF3 bootstrap replicates       1,000
```

## Principal output folders

```text
results/full_analysis/gb/
results/full_analysis/mosaic_orf/
results/full_analysis/orf3/
results/full_analysis/orf3_phylogeny/
results/full_analysis/compare_distances/
results/full_analysis/tree_mosaic_agreement/
results/full_analysis/mosaic_complexity_score_audit/
results/full_analysis/parameter_sensitivity/
results/full_analysis/orf_length_threshold_sensitivity/
results/full_analysis/entropy/
results/full_analysis/validation/
```

The full output inventory is written to:

```text
results/full_analysis/analysis_output_manifest.csv
```

## Validation modules

The `validation/` directory contains independent command-line checks for:

- agreement between whole-genome clusters and published species assignments;
- MDS Stress-1 and pairwise-distance reconstruction;
- barcode K selection, cross-seed stability, and Euclidean-versus-spherical geometry;
- unrooted ORF3 neighbor-joining bootstrap support;
- ORF-boundary enrichment sensitivity at 100, 200, and 400 bp;
- accession-level record screening and predicted-longest-ORF versus annotated-ORF3 comparison.

See `validation/README.md` for module-specific commands and outputs.

## Optional reference-panel projection

To project the combined 48-genome query set onto a broader reference panel:

```bash
python addons/addon_03_reference_panel_projection.py \
  --query_fasta "data/combined_core48.fasta" \
  --ref_fasta "path/to/reference_panel.fasta" \
  --k 4
```

Outputs:

```text
projection_nearest_reference.tsv
projection_token_summary.tsv
```

Matching sequence identifiers are excluded as self-hits when possible.

## Intended use and scope

The workflow is intended for post-sequencing comparative genomic analysis of complete or near-complete viral genomes. It can support reference placement, species-level comparison, nearest-reference annotation, and identification of genomes with unusual local barcode complexity.

It is not a direct field diagnostic assay and does not identify nucleotide-resolution recombination breakpoints. Breakpoint-oriented recombination analyses remain appropriate for testing specific recombination hypotheses. Epidemiological tracing additionally requires independently curated sampling dates, geographic locations, hosts, vectors, and, for mixed infections, haplotype-resolved viral genomes.

## Reproducibility

- Stochastic steps use explicit random seeds.
- Circular barcode analysis includes origin-spanning windows and the last-to-first barcode boundary.
- The public score uses barcode-derived components only; ORF–mosaic mismatch is reported separately.
- Input accession lists and published species labels are versioned in the repository.
- Release-specific source archives and commit identifiers can be used to reproduce the exact public code state.

## License

MIT License. See `LICENSE.txt`.
