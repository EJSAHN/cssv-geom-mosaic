#!/usr/bin/env python3
"""Run the corrected CSSV baseline analysis and validation suite.

The workflow uses circular barcode windows, circular barcode adjacency, and
entropy normalized by log2(K). It then runs published-species validation,
barcode-clustering diagnostics, ORF3 NJ bootstrap support, ORF-boundary
threshold sensitivity, MDS diagnostics, and accession/ORF3/EVE record audits.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def run(cmd: List[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[ERR] command failed ({proc.returncode}): {' '.join(cmd)}")


def py(repo: Path, rel: str, *args: str) -> List[str]:
    return [sys.executable, str(repo / rel), *map(str, args)]


def count_individual_records(folder: Path, suffixes: set[str]) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes and not p.name.lower().startswith("combined_"))


def write_manifest(root: Path) -> None:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rows.append({"relative_path": str(p.relative_to(root)), "size_bytes": p.stat().st_size})
    with open(root / "analysis_output_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes"])
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--input_dir", required=True, help="Folder containing exactly the 48 individual FASTA/GenBank records")
    ap.add_argument("--genbank_dir", default=None, help="Folder containing accession-level GenBank records for annotation audits")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--existing_alignment", default=None, help="Existing aligned ORF3 protein FASTA; avoids rerunning MAFFT")
    ap.add_argument("--species_table", default=None)
    ap.add_argument("--reference_panel_fasta", default=None, help="Optional broader panel for nearest-reference projection")
    ap.add_argument("--bootstrap_replicates", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_parameter_sensitivity", action="store_true")
    ap.add_argument("--skip_k_diagnostics", action="store_true")
    ap.add_argument("--skip_entropy", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    inp = Path(args.input_dir).resolve()
    out = Path(args.out_dir).resolve()
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        raise SystemExit(f"[ERR] input directory not found: {inp}")
    n_inputs = count_individual_records(inp, {".fa", ".fasta", ".fna", ".gb", ".gbk", ".genbank"})
    if n_inputs != 48:
        raise SystemExit(f"[ERR] Expected exactly 48 individual input files; found {n_inputs} in {inp}")

    species_table = Path(args.species_table).resolve() if args.species_table else repo / "resources" / "core48_published_species.tsv"
    gb_dir = out / "gb"
    mosaic_dir = out / "mosaic_orf"
    orfseq_dir = out / "orf3" / "orf_seqs"
    longest_dir = out / "orf3"
    phylo_dir = out / "orf3_phylogeny"
    compare_dir = out / "compare_distances"
    agreement_dir = out / "tree_mosaic_agreement"

    # Corrected circular baseline.
    run(py(repo, "pipeline/cssv_gb_pipeline.py",
           "--input_dir", str(inp), "--out_dir", str(gb_dir), "--k", "4",
           "--default_topology", "circular", "--do_windows", "--window", "250",
           "--step", "50", "--n_clusters", "8", "--random_state", "0"), repo)

    run(py(repo, "pipeline/cssv_mosaic_orf_analysis.py",
           "--input_dir", str(inp), "--window_assignments", str(gb_dir / "window_assignments.csv"),
           "--out_dir", str(mosaic_dir), "--circular", "--start_codons", "ATG",
           "--min_orf_aa", "300", "--switchpoint_mode", "start", "--min_run", "2",
           "--near_bp", "200", "--perm", "2000", "--seed", "0", "--plot_top", "0", "--no_plots"), repo)

    run(py(repo, "pipeline/cssv_extract_predicted_orf_seqs.py",
           "--input_dir", str(inp), "--predicted_orfs", str(mosaic_dir / "predicted_orfs.csv"),
           "--out_dir", str(orfseq_dir)), repo)
    run(py(repo, "pipeline/cssv_extract_longest_orfs.py",
           "--predicted_orfs_csv", str(mosaic_dir / "predicted_orfs.csv"),
           "--orf_faa", str(orfseq_dir / "predicted_orfs.faa"),
           "--orf_fna", str(orfseq_dir / "predicted_orfs.fna"),
           "--out_dir", str(longest_dir), "--top_n", "1"), repo)

    existing_alignment = Path(args.existing_alignment).resolve() if args.existing_alignment else None
    if existing_alignment is not None:
        if not existing_alignment.is_file():
            raise SystemExit(
                f"[ERR] --existing_alignment must point to an aligned protein FASTA file, not a directory or missing path: {existing_alignment}"
            )
        faa_for_tree = existing_alignment
        aligner = "none"
        print(f"[INFO] Reusing existing ORF3 alignment: {existing_alignment}")
    else:
        faa_for_tree = longest_dir / "longest_orfs.faa"
        aligner = "auto"
        print("[INFO] No existing alignment supplied; MAFFT will be attempted natively and then through WSL.")
    run(py(repo, "pipeline/cssv_orf_msa_tree.py",
           "--faa", str(faa_for_tree), "--out_dir", str(phylo_dir),
           "--aligner", aligner, "--gap_cutoff", "0.5", "--no_plots"), repo)

    orf_dist = phylo_dir / "pairwise_identity_distance.csv"
    trimmed_alignment = phylo_dir / "alignment.trim_gap0.50.faa"
    run(py(repo, "pipeline/cssv_compare_distances.py",
           "--matrix_a", str(gb_dir / "k4_cosine_distance.csv"),
           "--matrix_b", str(orf_dist), "--b_split", "|", "--out_dir", str(compare_dir),
           "--method", "spearman", "--perm", "5000", "--seed", "0"), repo)
    run(py(repo, "scripts/cssv_switchpoint_postprocess.py",
           "--switchpoints", str(mosaic_dir / "switchpoints.csv"),
           "--genome_summary", str(gb_dir / "genome_summary.csv"),
           "--out_dir", str(out / "switchpoint_post"), "--bins", "120"), repo)
    run(py(repo, "pipeline/cssv_tree_mosaic_agreement.py",
           "--orf_dist", str(orf_dist), "--orf_name_split", "|",
           "--window_assignments", str(gb_dir / "window_assignments.csv"),
           "--out_dir", str(agreement_dir), "--k_orf", "8", "--mosaic_k", "8",
           "--min_purity", "0.6", "--top_n", "20", "--tree_newick", str(phylo_dir / "nj_tree.newick")), repo)
    run(py(repo, "analysis_checks/cssv_mosaic_complexity_score_audit.py",
           "--merged", str(agreement_dir / "mosaic_orf_merged_per_genome.csv"),
           "--out_dir", str(out / "mosaic_complexity_score_audit"), "--top_n", "10", "--mosaic_k", "8"), repo)

    if not args.skip_entropy:
        entropy_dir = out / "entropy"
        run(py(repo, "pipeline/cssv_entropy_analysis.py",
               "--input_dir", str(inp), "--out_dir", str(entropy_dir),
               "--window", "200", "--step", "50", "--k", "4", "--no_plots"), repo)
        run(py(repo, "pipeline/cssv_compare_distances.py",
               "--matrix_a", str(gb_dir / "k4_cosine_distance.csv"),
               "--matrix_b", str(entropy_dir / "jsd_distance_matrix.csv"),
               "--out_dir", str(entropy_dir / "compare_jsd_vs_cosine"),
               "--method", "spearman", "--perm", "5000", "--seed", "0"), repo)
        run(py(repo, "pipeline/cssv_compare_distances.py",
               "--matrix_a", str(entropy_dir / "jsd_distance_matrix.csv"),
               "--matrix_b", str(orf_dist), "--b_split", "|",
               "--out_dir", str(entropy_dir / "compare_jsd_vs_orf3"),
               "--method", "spearman", "--perm", "5000", "--seed", "0"), repo)
        run(py(repo, "scripts/cssv_entropy_mosaic_ranked_test.py",
               "--entropy_profiles", str(entropy_dir / "entropy_profiles.csv"),
               "--ranked_genomes", str(agreement_dir / "mosaic_ranked_genomes.csv"),
               "--out_dir", str(entropy_dir / "mosaic_ranked_entropy_test"),
               "--top_n", "10", "--perm", "10000", "--seed", "0", "--no_plots"), repo)
        run(py(repo, "scripts/cssv_entropy_tail_metrics.py",
               "--entropy_profiles", str(entropy_dir / "entropy_profiles.csv"),
               "--ranked_genomes", str(agreement_dir / "mosaic_ranked_genomes.csv"),
               "--out_dir", str(entropy_dir / "mosaic_ranked_entropy_tail_test"),
               "--top_n", "10", "--perm", "10000", "--seed", "0", "--no_plots"), repo)

    if not args.skip_parameter_sensitivity:
        run(py(repo, "analysis_checks/cssv_parameter_sensitivity.py",
               "--repo", str(repo), "--input_dir", str(inp), "--orf_dist", str(orf_dist),
               "--out_dir", str(out / "parameter_sensitivity")), repo)
        run(py(repo, "analysis_checks/cssv_orf_threshold_sensitivity.py",
               "--repo", str(repo), "--input_dir", str(inp),
               "--window_assignments", str(gb_dir / "window_assignments.csv"),
               "--out_dir", str(out / "orf_length_threshold_sensitivity"),
               "--thresholds", "200,300,400", "--near_bp", "200", "--perm", "2000", "--seed", "0"), repo)

    # External and diagnostic validation.
    validation_root = out / "validation"
    run(py(repo, "validation/cssv_external_species_validation.py",
           "--distance_matrix", str(gb_dir / "k4_cosine_distance.csv"),
           "--species_table", str(species_table),
           "--out_dir", str(validation_root / "published_species")), repo)
    run(py(repo, "validation/cssv_mds_diagnostics.py",
           "--distance_matrix", str(gb_dir / "k4_cosine_distance.csv"),
           "--embedding", str(gb_dir / "genome_embedding.csv"),
           "--out_dir", str(validation_root / "mds")), repo)
    run(py(repo, "validation/cssv_orf_boundary_threshold_sensitivity.py",
           "--genome_summary", str(gb_dir / "genome_summary.csv"),
           "--predicted_orfs", str(mosaic_dir / "predicted_orfs.csv"),
           "--switchpoints", str(mosaic_dir / "switchpoints.csv"),
           "--out_dir", str(validation_root / "orf_boundary_thresholds"),
           "--thresholds", "100,200,400", "--perm", "2000", "--seed", "0"), repo)
    run(py(repo, "validation/cssv_orf3_nj_bootstrap.py",
           "--alignment", str(trimmed_alignment), "--out_dir", str(validation_root / "orf3_bootstrap"),
           "--replicates", str(args.bootstrap_replicates), "--seed", "0", "--name_split", "|"), repo)
    if not args.skip_k_diagnostics:
        run(py(repo, "validation/cssv_barcode_clustering_diagnostics.py",
               "--repo", str(repo), "--input_dir", str(inp),
               "--out_dir", str(validation_root / "barcode_clustering"),
               "--kmer_k", "4", "--window", "250", "--step", "50",
               "--k_values", "2-12", "--seeds", "0,1,2,3,4,5,6,7,8,9", "--baseline_k", "8",
               "--baseline_assignments", str(gb_dir / "window_assignments.csv")), repo)

    genbank_dir = Path(args.genbank_dir).resolve() if args.genbank_dir else None
    if genbank_dir and genbank_dir.exists() and count_individual_records(genbank_dir, {".gb", ".gbk", ".genbank"}) >= 48:
        run(py(repo, "validation/cssv_orf3_eve_record_audit.py",
               "--genbank_dir", str(genbank_dir),
               "--predicted_orfs", str(mosaic_dir / "predicted_orfs.csv"),
               "--accessions", str(repo / "ACCESSIONS.txt"),
               "--species_table", str(species_table),
               "--out_dir", str(validation_root / "accession_orf3_eve_audit")), repo)
    else:
        print("[WARN] Complete accession-level GenBank folder not supplied; ORF3/EVE record audit was skipped.")

    if args.reference_panel_fasta:
        panel = Path(args.reference_panel_fasta).resolve()
        combined_query = inp.parent / "combined_core48.fasta"
        if panel.exists() and combined_query.exists():
            run(py(repo, "addons/addon_03_reference_panel_projection.py",
                   "--query_fasta", str(combined_query), "--ref_fasta", str(panel), "--k", "4"), out)
        else:
            print("[WARN] Reference projection skipped; panel or combined query FASTA is missing.")

    write_manifest(out)
    print("\n[DONE] Corrected baseline and validation suite completed.")
    print(f"[DONE] Output root: {out}")


if __name__ == "__main__":
    main()
