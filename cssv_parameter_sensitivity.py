#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cssv_parameter_sensitivity.py

Parameter-sensitivity utility for the CSSV genome-geometry/mosaic-barcode pipeline.

This script separates two issues:
  1) Mosaic-barcode parameter sensitivity: k, window, step, and window-label K
     are varied one at a time while ORF-cluster K is held fixed at baseline.
  2) ORF-cluster K sensitivity: ORF clustering K is varied while the baseline
     mosaic barcode assignments are held fixed.

This avoids confounding mosaic-label K with ORF-cluster K.

Inputs
------
  --repo       Path to repository root containing pipeline/*.py
  --input_dir Raw GenBank/FASTA input folder
  --orf_dist  ORF3 pairwise_identity_distance.csv from the core pipeline

Outputs
-------
  sensitivity_summary.csv
  sensitivity_topN_by_config.csv
  sensitivity_topN_overlap_matrix.csv

No hardcoded directories.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def csv_int_list(x: str) -> List[int]:
    return [int(v.strip()) for v in str(x).split(",") if v.strip()]


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[ERR] command failed ({proc.returncode}): {' '.join(cmd)}")


def safe_mosaic_name(k: int, window: int, step: int, mosaic_K: int, orf_K: int) -> str:
    return f"mosaic_k{k}_w{window}_s{step}_K{mosaic_K}_orfK{orf_K}"


def safe_orfK_name(orf_K: int) -> str:
    return f"baseline_mosaic_orfK{orf_K}"


def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def mosaic_configs(args) -> List[Tuple[int, int, int, int]]:
    baseline = (args.baseline_k, args.baseline_window, args.baseline_step, args.baseline_mosaic_clusters)
    configs = {baseline}
    for k in csv_int_list(args.k_values):
        configs.add((k, args.baseline_window, args.baseline_step, args.baseline_mosaic_clusters))
    for w in csv_int_list(args.window_values):
        configs.add((args.baseline_k, w, args.baseline_step, args.baseline_mosaic_clusters))
    for s in csv_int_list(args.step_values):
        configs.add((args.baseline_k, args.baseline_window, s, args.baseline_mosaic_clusters))
    for K in csv_int_list(args.mosaic_cluster_values):
        configs.add((args.baseline_k, args.baseline_window, args.baseline_step, K))
    return sorted(configs)


def read_scores(path: Path) -> pd.DataFrame:
    """Read per-genome ranking table.

    The score column is expected to be ``mosaic_complexity_score``.
    """
    df = pd.read_csv(path)
    if "genome" not in df.columns or "mosaic_complexity_score" not in df.columns:
        raise ValueError(
            f"Expected 'genome' and 'mosaic_complexity_score' columns in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    df["mosaic_complexity_score"] = pd.to_numeric(df["mosaic_complexity_score"], errors="coerce")
    return df.sort_values("mosaic_complexity_score", ascending=False).reset_index(drop=True)


def run_gb(repo: Path, gb_script: Path, input_dir: Path, gb_dir: Path, k: int, window: int, step: int, mosaic_K: int, dry_run: bool):
    gb_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(gb_script),
        "--input_dir", str(input_dir),
        "--out_dir", str(gb_dir),
        "--k", str(k),
        "--default_topology", "circular",
        "--do_windows",
        "--window", str(window),
        "--step", str(step),
        "--n_clusters", str(mosaic_K),
    ]
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
    elif not (gb_dir / "window_assignments.csv").exists():
        run_cmd(cmd, cwd=repo)
    else:
        print(f"[SKIP] Existing {gb_dir / 'window_assignments.csv'}")


def run_agreement(repo: Path, agree_script: Path, orf_dist: Path, window_assignments: Path, agree_dir: Path,
                  orf_K: int, mosaic_K: int, top_n: int, orf_name_split: str, dry_run: bool):
    agree_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(agree_script),
        "--orf_dist", str(orf_dist),
        "--orf_name_split", str(orf_name_split),
        "--window_assignments", str(window_assignments),
        "--out_dir", str(agree_dir),
        "--k_orf", str(orf_K),
        "--mosaic_k", str(mosaic_K),
        "--min_purity", "0.6",
        "--top_n", str(top_n),
    ]
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
    elif not (agree_dir / "mosaic_orf_merged_per_genome.csv").exists():
        run_cmd(cmd, cwd=repo)
    else:
        print(f"[SKIP] Existing {agree_dir / 'mosaic_orf_merged_per_genome.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parameter sensitivity with mosaic-K and ORF-K separated.")
    ap.add_argument("--repo", default=".", help="Repository root containing pipeline/*.py")
    ap.add_argument("--input_dir", required=True, help="Raw GenBank/FASTA input directory")
    ap.add_argument("--orf_dist", required=True, help="ORF3 pairwise_identity_distance.csv")
    ap.add_argument("--out_dir", default="results/parameter_sensitivity")
    ap.add_argument("--orf_name_split", default="|", help="Passed to cssv_tree_mosaic_agreement.py")
    ap.add_argument("--top_n", type=int, default=10)

    ap.add_argument("--baseline_k", type=int, default=4)
    ap.add_argument("--baseline_window", type=int, default=250)
    ap.add_argument("--baseline_step", type=int, default=50)
    ap.add_argument("--baseline_mosaic_clusters", type=int, default=8)
    ap.add_argument("--baseline_orf_clusters", type=int, default=8)

    ap.add_argument("--k_values", default="3,4,5")
    ap.add_argument("--window_values", default="200,250,300")
    ap.add_argument("--step_values", default="50,100")
    ap.add_argument("--mosaic_cluster_values", default="6,7,8,9,10")
    ap.add_argument("--orf_cluster_values", default="6,7,8,9,10")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    input_dir = Path(args.input_dir).resolve()
    orf_dist = Path(args.orf_dist).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gb_script = repo / "pipeline" / "cssv_gb_pipeline.py"
    agree_script = repo / "pipeline" / "cssv_tree_mosaic_agreement.py"
    for p, label in [(gb_script, "gb pipeline"), (agree_script, "agreement script"), (input_dir, "input_dir"), (orf_dist, "orf_dist")]:
        if not p.exists():
            raise SystemExit(f"[ERR] missing {label}: {p}")

    score_tables: Dict[str, pd.DataFrame] = {}
    rows = []
    top_rows = []

    # 1) Mosaic parameter sensitivity, ORF K fixed.
    for k, window, step, mosaic_K in mosaic_configs(args):
        orf_K = args.baseline_orf_clusters
        name = safe_mosaic_name(k, window, step, mosaic_K, orf_K)
        cfg_dir = out_dir / name
        gb_dir = cfg_dir / "gb"
        agree_dir = cfg_dir / "tree_mosaic_agreement"
        run_gb(repo, gb_script, input_dir, gb_dir, k, window, step, mosaic_K, args.dry_run)
        run_agreement(repo, agree_script, orf_dist, gb_dir / "window_assignments.csv", agree_dir, orf_K, mosaic_K, args.top_n, args.orf_name_split, args.dry_run)
        if args.dry_run:
            continue
        scores = read_scores(agree_dir / "mosaic_orf_merged_per_genome.csv")
        scores["rank"] = np.arange(1, len(scores) + 1)
        score_tables[name] = scores[["genome", "rank", "mosaic_complexity_score"]].copy()
        top = scores.head(args.top_n)["genome"].tolist()
        rows.append({
            "analysis_type": "mosaic_parameter_sensitivity_orfK_fixed",
            "config": name, "k": k, "window": window, "step": step,
            "mosaic_K": mosaic_K, "orf_K": orf_K,
            "topN_genomes": ";".join(top),
            "score_rank1": float(scores.iloc[0]["mosaic_complexity_score"]),
            "score_rankN": float(scores.iloc[min(args.top_n, len(scores))-1]["mosaic_complexity_score"]),
        })
        for _, r in scores.head(args.top_n).iterrows():
            top_rows.append({"analysis_type": "mosaic_parameter_sensitivity_orfK_fixed", "config": name, "rank": int(r["rank"]), "genome": r["genome"], "mosaic_complexity_score": float(r["mosaic_complexity_score"])})

    # 2) ORF K sensitivity, baseline mosaic fixed.
    baseline_mosaic_name = safe_mosaic_name(args.baseline_k, args.baseline_window, args.baseline_step,
                                            args.baseline_mosaic_clusters, args.baseline_orf_clusters)
    baseline_gb_dir = out_dir / baseline_mosaic_name / "gb"
    if not args.dry_run and not (baseline_gb_dir / "window_assignments.csv").exists():
        raise SystemExit(f"[ERR] baseline window assignments missing: {baseline_gb_dir / 'window_assignments.csv'}")
    for orf_K in csv_int_list(args.orf_cluster_values):
        name = safe_orfK_name(orf_K)
        agree_dir = out_dir / name / "tree_mosaic_agreement"
        run_agreement(repo, agree_script, orf_dist, baseline_gb_dir / "window_assignments.csv", agree_dir,
                      orf_K, args.baseline_mosaic_clusters, args.top_n, args.orf_name_split, args.dry_run)
        if args.dry_run:
            continue
        scores = read_scores(agree_dir / "mosaic_orf_merged_per_genome.csv")
        scores["rank"] = np.arange(1, len(scores) + 1)
        score_tables[name] = scores[["genome", "rank", "mosaic_complexity_score"]].copy()
        top = scores.head(args.top_n)["genome"].tolist()
        rows.append({
            "analysis_type": "orf_cluster_K_sensitivity_baseline_mosaic_fixed",
            "config": name,
            "k": args.baseline_k, "window": args.baseline_window, "step": args.baseline_step,
            "mosaic_K": args.baseline_mosaic_clusters, "orf_K": orf_K,
            "topN_genomes": ";".join(top),
            "score_rank1": float(scores.iloc[0]["mosaic_complexity_score"]),
            "score_rankN": float(scores.iloc[min(args.top_n, len(scores))-1]["mosaic_complexity_score"]),
        })
        for _, r in scores.head(args.top_n).iterrows():
            top_rows.append({"analysis_type": "orf_cluster_K_sensitivity_baseline_mosaic_fixed", "config": name, "rank": int(r["rank"]), "genome": r["genome"], "mosaic_complexity_score": float(r["mosaic_complexity_score"])})

    if args.dry_run:
        return

    baseline_name = baseline_mosaic_name
    if baseline_name not in score_tables:
        raise SystemExit(f"[ERR] baseline config not found: {baseline_name}")
    baseline = score_tables[baseline_name].copy().sort_values("rank")
    baseline_top = baseline.head(args.top_n)["genome"].tolist()
    baseline_scores = baseline.set_index("genome")["mosaic_complexity_score"]

    summary = pd.DataFrame(rows)
    ov = []
    for name, tab in score_tables.items():
        top = tab.sort_values("rank").head(args.top_n)["genome"].tolist()
        common = sorted(set(baseline_scores.index) & set(tab["genome"]))
        rho = np.nan
        if spearmanr is not None and len(common) >= 3:
            rho = float(spearmanr(baseline_scores.loc[common], tab.set_index("genome").loc[common, "mosaic_complexity_score"]).correlation)
        ov.append({
            "config": name,
            "topN_overlap_count_vs_baseline": len(set(baseline_top) & set(top)),
            "topN_jaccard_vs_baseline": jaccard(baseline_top, top),
            "spearman_score_vs_baseline": rho,
        })
    summary = summary.merge(pd.DataFrame(ov), on="config", how="left")
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    pd.DataFrame(top_rows).to_csv(out_dir / "sensitivity_topN_by_config.csv", index=False)

    names = list(score_tables.keys())
    top_sets = {n: score_tables[n].sort_values("rank").head(args.top_n)["genome"].tolist() for n in names}
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a, b in itertools.product(names, names):
        mat.loc[a, b] = jaccard(top_sets[a], top_sets[b])
    mat.to_csv(out_dir / "sensitivity_topN_overlap_matrix.csv")


    print(f"[OK] Wrote sensitivity outputs to {out_dir}")
    print("[OK] Mosaic-parameter sensitivity kept ORF-cluster K fixed at baseline.")
    print("[OK] ORF-cluster K sensitivity was run separately using baseline mosaic assignments.")


if __name__ == "__main__":
    main()
