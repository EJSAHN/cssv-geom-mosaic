# cssv-geom-mosaic

Alignment-free genome geometry and mosaic (chimera) detection for cacao swollen shoot virus (CSSV), cross-validated against ORF3 phylogeny.

This repository provides a reproducible, command-line pipeline to:
1) compute alignment-free genome distances (k-mer cosine),
2) build sliding-window mosaic barcodes,
3) predict ORFs from sequence only (no GenBank CDS required),
4) derive ORF3 (longest ORF) protein phylogeny distances and NJ tree,
5) quantify concordance between genome-scale distances and ORF3 distances,
6) rank chimera candidates and generate publication-ready summary tables/figures.

## Data (NCBI)
Raw genome sequences are NOT included in this repository.  
Please download the 48 CSSV genomes from NCBI using the accession list in:

- `data/ACCESSIONS.txt`

You may download sequences as `.gb/.gbk` (preferred) or `.fasta`.  
Place downloaded files into a single input folder, e.g.:

- `data/raw/`

> Important: The pipeline can parse multi-record GenBank/FASTA files, but **one genome per file** is recommended for clarity.

## Installation

### Option A: Conda (recommended)
```bash
conda create -n cssv_geom python=3.11 -y
conda activate cssv_geom
pip install -r requirements.txt
# cssv-geom-mosaic
Alignment-free genome geometry and mosaic chimera detection for cacao swollen shoot virus (CSSV), cross-validated against ORF3 phylogeny.

### Option B: pip
pip install -r requirements.txt


### Quickstart (end-to-end)
Assume:
-raw genomes in: data/raw/
-outputs in: results/

1) Genome k-mer distances + MDS embedding + window barcodes
python pipeline/cssv_gb_pipeline.py \
  --input_dir "data/raw" \
  --out_dir "results/gb" \
  --k 4 \
  --default_topology circular \
  --do_windows \
  --window 250 \
  --step 50 \
  --n_clusters 8


Key outputs:
-results/gb/genome_summary.csv
-results/gb/k4_cosine_distance.csv
-results/gb/genome_embedding.csv
-results/gb/window_assignments.csv

2) Mosaic switchpoints + ORF prediction + ORF-boundary null test
python pipeline/cssv_mosaic_orf_analysis.py \
  --input_dir "data/raw" \
  --window_assignments "results/gb/window_assignments.csv" \
  --out_dir "results/mosaic_orf" \
  --circular \
  --start_codons ATG \
  --min_orf_aa 300 \
  --switchpoint_mode start \
  --min_run 2 \
  --near_bp 200 \
  --perm 2000 \
  --plot_top 10


Key outputs:
-results/mosaic_orf/predicted_orfs.csv
-results/mosaic_orf/switchpoints.csv
-results/mosaic_orf/switchpoint_orf_distances.csv
-results/mosaic_orf/enrichment_summary.csv
-results/mosaic_orf/null_fracs.npy

3) Extract predicted ORF nucleotide/protein sequences
python pipeline/cssv_extract_predicted_orf_seqs.py \
  --input_dir "data/raw" \
  --predicted_orfs "results/mosaic_orf/predicted_orfs.csv" \
  --out_dir "results/mosaic_orf/orf_seqs"


Outputs:
-results/mosaic_orf/orf_seqs/predicted_orfs.fna
-results/mosaic_orf/orf_seqs/predicted_orfs.faa
-results/mosaic_orf/orf_seqs/predicted_orfs_extracted.csv

4) Select the longest ORF per genome (ORF3 proxy)
python pipeline/cssv_extract_longest_orfs.py \
  --predicted_orfs_csv "results/mosaic_orf/predicted_orfs.csv" \
  --orf_faa "results/mosaic_orf/orf_seqs/predicted_orfs.faa" \
  --orf_fna "results/mosaic_orf/orf_seqs/predicted_orfs.fna" \
  --out_dir "results/orf3" \
  --top_n 1

Outputs:
-results/orf3/longest_orfs.faa
-results/orf3/longest_orfs.fna
-results/orf3/longest_orfs.csv

### MAFFT step (run OUTSIDE the Python pipeline)
We intentionally keep MAFFT outside the main Python pipeline because MAFFT installation and permissions can differ across platforms (Windows/WSL/Linux/macOS).

A) Run MAFFT (recommended via WSL/Linux/macOS)

Align the ORF3 proteins:
mafft --auto --anysymbol "results/orf3/longest_orfs.faa" > "results/orf3/alignment.mafft.faa"


Sanity check:
-results/orf3/alignment.mafft.faa must be non-empty and start with >.

If you run MAFFT in WSL and see permission issues, run it as root and redirect stderr:

sudo mafft --auto --anysymbol "results/orf3/longest_orfs.faa" \
  1> "results/orf3/alignment.mafft.faa" \
  2> "results/orf3/mafft.stderr.log"

B) Continue the pipeline using the precomputed alignment

Now compute trimmed alignment, ORF distance matrix, and NJ tree without running MAFFT inside Python:

python pipeline/cssv_orf_msa_tree.py \
  --faa "results/orf3/alignment.mafft.faa" \
  --out_dir "results/orf3_phylogeny" \
  --aligner none \
  --gap_cutoff 0.5


Outputs:

-results/orf3_phylogeny/alignment.trim_gap0.50.faa
-results/orf3_phylogeny/pairwise_identity_distance.csv
-results/orf3_phylogeny/nj_tree.newick
-results/orf3_phylogeny/nj_tree.pdf
-results/orf3_phylogeny/pairwise_distance_heatmap.pdf

Concordance tests and chimera ranking
5) Distance–distance concordance (Mantel-style permutation)
python pipeline/cssv_compare_distances.py \
  --matrix_a "results/gb/k4_cosine_distance.csv" \
  --matrix_b "results/orf3_phylogeny/pairwise_identity_distance.csv" \
  --b_split "|" \
  --out_dir "results/compare_distances" \
  --method spearman \
  --perm 5000 \
  --seed 0 \
  --plot

6) ORF–mosaic agreement + chimera candidates
python pipeline/cssv_tree_mosaic_agreement.py \
  --orf_dist "results/orf3_phylogeny/pairwise_identity_distance.csv" \
  --orf_name_split "|" \
  --window_assignments "results/gb/window_assignments.csv" \
  --out_dir "results/tree_mosaic_agreement" \
  --k_orf 8 \
  --min_purity 0.6 \
  --top_n 20 \
  --tree_newick "results/orf3_phylogeny/nj_tree.newick"


Key outputs:
-results/tree_mosaic_agreement/chimera_candidates.csv
-results/tree_mosaic_agreement/mosaic_orf_merged_per_genome.csv
-results/tree_mosaic_agreement/agreement_metrics.csv
-results/tree_mosaic_agreement/contingency_orf_vs_mosaic.csv

---

## Paper-ready figures and Supplementary Data (Figure 1–6, Table 1, S1.xlsx)

After you have generated all core results (gb pipeline → mosaic/orf → ORF3 phylogeny → concordance → chimera ranking),
you can build final paper-ready figures (PDF vector + PNG 300 dpi), Table 1 (Excel), and Supplementary Data S1 (single multi-sheet Excel).

Example (Windows; adjust paths as needed):

```bat
python paper\cssv_make_paper_package.py --gb_dir "gb_results" --mosaic_orf_dir "gb_results\mosaic_orf" --orf3_dir "gb_results\mosaic_orf\longest_orfs" --orf3_phylogeny_dir "gb_results\mosaic_orf\longest_orfs\phylogeny" --compare_dir "gb_results\distance_compare" --agreement_dir "gb_results\tree_mosaic_agreement" --switchpoint_post_dir "gb_results\switchpoint_post" --out_dir "paper_outputs" --top_n 10 --panel_case upper --dpi 300

Reproducibility notes

All steps are deterministic given fixed random seeds (where used).

Raw inputs must match the accessions listed in data/ACCESSIONS.txt.
Raw genomes are downloaded from NCBI using the accessions in data/ACCESSIONS.txt



