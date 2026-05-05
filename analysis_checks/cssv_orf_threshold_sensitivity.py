#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cssv_orf_threshold_sensitivity.py

ORF-threshold sensitivity utility for the CSSV genome-geometry/mosaic-barcode pipeline.

Purpose
-------
Audits the 300-aa ORF threshold by re-running the
ORF/switchpoint boundary analysis at multiple thresholds and summarizing whether
boundary-enrichment conclusions change.

Inputs
------
  --repo                Path to repository root containing pipeline/*.py
  --input_dir            Raw GenBank/FASTA input folder
  --window_assignments   Baseline window_assignments.csv

Outputs
-------
  orf_threshold_sensitivity_summary.csv
  Individual threshold folders with predicted_orfs.csv, switchpoints.csv,
  switchpoint_orf_distances.csv, enrichment_summary.csv, and null_fracs.npy.

The underlying core ORF script can create plot files, so this wrapper removes any plot folder after each threshold run. Final retained outputs are tabular/array files only.

No hardcoded directories.
"""
from __future__ import annotations

import argparse
import subprocess
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def csv_int_list(x: str) -> List[int]:
    return [int(v.strip()) for v in str(x).split(",") if v.strip()]


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[ERR] command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def read_enrichment(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def summarize_orfs(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {"n_orfs": 0, "n_genomes_with_orf": 0, "median_orf_aa": np.nan, "max_orf_aa": np.nan}
    genome_col = "genome" if "genome" in df.columns else ("name" if "name" in df.columns else None)
    aa_col = None
    for c in ["aa_len", "length_aa", "protein_length", "orf_aa_len", "aa_length"]:
        if c in df.columns:
            aa_col = c
            break
    out = {"n_orfs": len(df)}
    if genome_col:
        out["n_genomes_with_orf"] = int(df[genome_col].nunique())
    if aa_col:
        aa = pd.to_numeric(df[aa_col], errors="coerce")
        out["median_orf_aa"] = float(np.nanmedian(aa))
        out["max_orf_aa"] = float(np.nanmax(aa))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit ORF min-aa threshold effects on ORF-boundary enrichment.")
    ap.add_argument("--repo", default=".", help="Repository root containing pipeline/*.py")
    ap.add_argument("--input_dir", required=True, help="Raw GenBank/FASTA input directory")
    ap.add_argument("--window_assignments", required=True, help="Baseline window_assignments.csv")
    ap.add_argument("--out_dir", default="results/orf_threshold_sensitivity")
    ap.add_argument("--thresholds", default="200,300,400", help="Comma-separated min ORF aa thresholds")
    ap.add_argument("--near_bp", type=int, default=200)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    input_dir = Path(args.input_dir).resolve()
    window_assignments = Path(args.window_assignments).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    script = repo / "pipeline" / "cssv_mosaic_orf_analysis.py"
    if not script.exists():
        raise SystemExit(f"[ERR] missing script: {script}")
    if not input_dir.exists():
        raise SystemExit(f"[ERR] missing input_dir: {input_dir}")
    if not window_assignments.exists():
        raise SystemExit(f"[ERR] missing window_assignments: {window_assignments}")

    rows = []
    for thr in csv_int_list(args.thresholds):
        td = out_dir / f"minorf{thr}aa"
        td.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(script),
            "--input_dir", str(input_dir),
            "--window_assignments", str(window_assignments),
            "--out_dir", str(td),
            "--circular",
            "--start_codons", "ATG",
            "--min_orf_aa", str(thr),
            "--switchpoint_mode", "start",
            "--min_run", "2",
            "--near_bp", str(args.near_bp),
            "--perm", str(args.perm),
            "--seed", str(args.seed),
            "--plot_top", "0",
        ]
        if args.dry_run:
            print("[DRY-RUN]", " ".join(cmd))
            continue
        if not (td / "enrichment_summary.csv").exists():
            run_cmd(cmd, cwd=repo)
        else:
            print(f"[SKIP] Existing {td / 'enrichment_summary.csv'}")
        plots_dir = td / "plots"
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
            print(f"[OK] Removed plot outputs from {plots_dir}")

        row = {"min_orf_aa": thr}
        row.update(summarize_orfs(td / "predicted_orfs.csv"))
        enrich = read_enrichment(td / "enrichment_summary.csv")
        # Prefix enrichment columns to avoid name conflicts.
        row.update({f"enrichment_{k}": v for k, v in enrich.items()})
        rows.append(row)

    if not args.dry_run:
        pd.DataFrame(rows).to_csv(out_dir / "orf_threshold_sensitivity_summary.csv", index=False)
        print("[OK] Wrote", out_dir / "orf_threshold_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
