#!/usr/bin/env python3
"""
Compare entropy statistics (mean/variance) between Chimera Top-N and others.

Inputs:
- entropy_profiles.csv from cssv_entropy_analysis.py
  required columns: genome, start, entropy
- chimera_candidates.csv from cssv_tree_mosaic_agreement.py
  required column: genome

Outputs:
- per_genome_entropy_stats.csv
- permutation_tests_summary.csv
- plots/entropy_mean_boxplot.pdf + .png (300 dpi)
- plots/entropy_variance_boxplot.pdf + .png (300 dpi)

Usage (Windows):
python scripts/cssv_entropy_chimera_test.py ^
  --entropy_profiles "results/entropy/entropy_profiles.csv" ^
  --chimera_candidates "gb_results/tree_mosaic_agreement/chimera_candidates.csv" ^
  --out_dir "results/entropy/chimera_entropy_test" ^
  --top_n 10 ^
  --perm 10000 ^
  --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def tip_to_accession(x: str) -> str:
    s = str(x).strip()
    s = s.split("|")[0]
    s = s.split("__dup")[0]
    return s


def permutation_test_two_sided(
    x: np.ndarray,
    y: np.ndarray,
    stat: str = "mean",
    perm: int = 10000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Two-sided permutation test on difference in statistic between groups.
    Returns (observed_diff, p_value)
    diff = stat(x) - stat(y)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")

    if stat == "mean":
        f = np.mean
    elif stat == "median":
        f = np.median
    else:
        raise ValueError("stat must be mean or median")

    obs = float(f(x) - f(y))

    pool = np.concatenate([x, y])
    n_x = x.size

    cnt = 0
    for _ in range(int(perm)):
        idx = rng.permutation(pool.size)
        xp = pool[idx[:n_x]]
        yp = pool[idx[n_x:]]
        dp = float(f(xp) - f(yp))
        if abs(dp) >= abs(obs):
            cnt += 1

    p = (1.0 + cnt) / (perm + 1.0)
    return obs, float(p)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    nx, ny = x.size, y.size
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)


def save_boxplot(
    out_pdf: Path,
    out_png: Path,
    data_top: np.ndarray,
    data_other: np.ndarray,
    ylabel: str,
    title: str,
    note_lines: list[str],
    dpi: int = 300,
) -> None:
    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(111)

    ax.boxplot([data_other, data_top], labels=["Others", "Chimera TopN"], showfliers=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Put annotation box at top-right inside axes
    ax.text(
        0.98,
        0.98,
        "\n".join(note_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.2),
    )

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy_profiles", required=True, help="entropy_profiles.csv")
    ap.add_argument("--chimera_candidates", required=True, help="chimera_candidates.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stat", choices=["mean", "median"], default="mean", help="Statistic for permutation test")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ent = pd.read_csv(args.entropy_profiles)
    chim = pd.read_csv(args.chimera_candidates)

    # Validate columns
    for col in ["genome", "start", "entropy"]:
        if col not in ent.columns:
            raise ValueError(f"entropy_profiles is missing column: {col}")
    if "genome" not in chim.columns:
        raise ValueError("chimera_candidates is missing column: genome")

    # Normalize genome names to accessions
    ent["genome"] = ent["genome"].astype(str).map(tip_to_accession)
    chim["genome"] = chim["genome"].astype(str).map(tip_to_accession)

    topN = chim.head(int(args.top_n))["genome"].tolist()
    top_set = set(topN)

    # Per-genome entropy stats
    ent["entropy"] = pd.to_numeric(ent["entropy"], errors="coerce")

    gstats = (
        ent.groupby("genome")["entropy"]
        .agg(
            n_windows="size",
            mean_entropy="mean",
            var_entropy=lambda x: float(np.nanvar(x.to_numpy(dtype=float), ddof=1)) if np.isfinite(x).sum() >= 2 else np.nan,
            std_entropy=lambda x: float(np.nanstd(x.to_numpy(dtype=float), ddof=1)) if np.isfinite(x).sum() >= 2 else np.nan,
        )
        .reset_index()
    )

    gstats["group"] = np.where(gstats["genome"].isin(top_set), "chimera_topN", "other")

    # Save per-genome table
    gstats_path = out_dir / "per_genome_entropy_stats.csv"
    gstats.to_csv(gstats_path, index=False)

    # Split arrays
    top_mean = gstats.loc[gstats["group"] == "chimera_topN", "mean_entropy"].to_numpy(dtype=float)
    oth_mean = gstats.loc[gstats["group"] == "other", "mean_entropy"].to_numpy(dtype=float)
    top_var = gstats.loc[gstats["group"] == "chimera_topN", "var_entropy"].to_numpy(dtype=float)
    oth_var = gstats.loc[gstats["group"] == "other", "var_entropy"].to_numpy(dtype=float)

    # Permutation tests (two-sided)
    diff_mean, p_mean = permutation_test_two_sided(top_mean, oth_mean, stat=args.stat, perm=args.perm, seed=args.seed)
    diff_var, p_var = permutation_test_two_sided(top_var, oth_var, stat=args.stat, perm=args.perm, seed=args.seed + 1)

    d_mean = cohen_d(top_mean, oth_mean)
    d_var = cohen_d(top_var, oth_var)

    summary = pd.DataFrame(
        [
            {
                "top_n": int(args.top_n),
                "perm": int(args.perm),
                "seed": int(args.seed),
                "test_stat": args.stat,
                "metric": "mean_entropy",
                "diff_top_minus_other": diff_mean,
                "p_two_sided": p_mean,
                "cohen_d": d_mean,
                "n_top": int(np.isfinite(top_mean).sum()),
                "n_other": int(np.isfinite(oth_mean).sum()),
            },
            {
                "top_n": int(args.top_n),
                "perm": int(args.perm),
                "seed": int(args.seed + 1),
                "test_stat": args.stat,
                "metric": "var_entropy",
                "diff_top_minus_other": diff_var,
                "p_two_sided": p_var,
                "cohen_d": d_var,
                "n_top": int(np.isfinite(top_var).sum()),
                "n_other": int(np.isfinite(oth_var).sum()),
            },
        ]
    )
    summary_path = out_dir / "permutation_tests_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plots
    save_boxplot(
        out_pdf=plots_dir / "entropy_mean_boxplot.pdf",
        out_png=plots_dir / "entropy_mean_boxplot.png",
        data_top=top_mean,
        data_other=oth_mean,
        ylabel="Mean window entropy (bits; A/C/G/T only)",
        title=f"Chimera Top{args.top_n} vs Others: mean entropy",
        note_lines=[
            rf"{args.stat} diff (Top−Other) = {diff_mean:.4g}",
            rf"$p$ = {p_mean:.4g}",
            rf"Cohen's d = {d_mean:.3f}" if np.isfinite(d_mean) else "Cohen's d = NA",
        ],
        dpi=args.dpi,
    )

    save_boxplot(
        out_pdf=plots_dir / "entropy_variance_boxplot.pdf",
        out_png=plots_dir / "entropy_variance_boxplot.png",
        data_top=top_var,
        data_other=oth_var,
        ylabel="Variance of window entropy (bits²)",
        title=f"Chimera Top{args.top_n} vs Others: entropy variance",
        note_lines=[
            rf"{args.stat} diff (Top−Other) = {diff_var:.4g}",
            rf"$p$ = {p_var:.4g}",
            rf"Cohen's d = {d_var:.3f}" if np.isfinite(d_var) else "Cohen's d = NA",
        ],
        dpi=args.dpi,
    )

    print("Done.")
    print(f"- Per-genome stats: {gstats_path}")
    print(f"- Permutation summary: {summary_path}")
    print(f"- Plots: {plots_dir}")


if __name__ == "__main__":
    main()
