#!/usr/bin/env python3
"""
Tail/peak entropy metrics for Chimera Top-N vs Others.

Goal:
- detect "local sparks/scars" (tail/peak metrics) even if global variance/mean is stable.

Inputs:
- entropy_profiles.csv (columns: genome,start,entropy)
- chimera_candidates.csv (column: genome)

Outputs:
- per_genome_entropy_tail_metrics.csv
- tail_permutation_tests_summary.csv
- plots/*.pdf + *.png (300 dpi)

Usage:
python scripts/cssv_entropy_tail_metrics.py ^
  --entropy_profiles "results/entropy/entropy_profiles.csv" ^
  --chimera_candidates "gb_results/tree_mosaic_agreement/chimera_candidates.csv" ^
  --out_dir "results/entropy/chimera_entropy_tail_test" ^
  --top_n 10 ^
  --perm 10000 ^
  --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, List

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
    diff = stat(x) - stat(y)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")

    f = np.mean if stat == "mean" else np.median
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


def boxplot_two_groups(
    out_pdf: Path,
    out_png: Path,
    others: np.ndarray,
    top: np.ndarray,
    ylabel: str,
    title: str,
    note_lines: List[str],
    dpi: int = 300,
) -> None:
    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(111)

    ax.boxplot([others, top], labels=["Others", "Chimera TopN"], showfliers=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

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


def per_genome_tail_metrics(ent: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tail/peak metrics per genome from window entropy values.
    """
    rows = []
    for genome, g in ent.groupby("genome"):
        x = g["entropy"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            continue

        p05 = float(np.quantile(x, 0.05))
        p25 = float(np.quantile(x, 0.25))
        p50 = float(np.quantile(x, 0.50))
        p75 = float(np.quantile(x, 0.75))
        p95 = float(np.quantile(x, 0.95))
        p99 = float(np.quantile(x, 0.99)) if x.size >= 100 else float(np.quantile(x, 0.95))  # fallback
        xmax = float(np.max(x))

        iqr = p75 - p25
        p95_minus_p05 = p95 - p05
        max_minus_median = xmax - p50
        tail_excess_95 = p95 - p50
        tail_excess_99 = p99 - p50

        rows.append(
            dict(
                genome=genome,
                n_windows=int(x.size),
                mean=float(np.mean(x)),
                var=float(np.var(x, ddof=1)) if x.size >= 2 else float("nan"),
                p05=p05,
                p25=p25,
                p50=p50,
                p75=p75,
                p95=p95,
                p99=p99,
                max=xmax,
                iqr=iqr,
                p95_minus_p05=p95_minus_p05,
                max_minus_median=max_minus_median,
                tail_excess_95=tail_excess_95,
                tail_excess_99=tail_excess_99,
            )
        )

    return pd.DataFrame(rows).sort_values("genome").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy_profiles", required=True)
    ap.add_argument("--chimera_candidates", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stat", choices=["mean", "median"], default="mean")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ent = pd.read_csv(args.entropy_profiles)
    chim = pd.read_csv(args.chimera_candidates)

    for c in ["genome", "start", "entropy"]:
        if c not in ent.columns:
            raise ValueError(f"entropy_profiles missing column: {c}")
    if "genome" not in chim.columns:
        raise ValueError("chimera_candidates missing column: genome")

    ent["genome"] = ent["genome"].astype(str).map(tip_to_accession)
    ent["entropy"] = pd.to_numeric(ent["entropy"], errors="coerce")
    chim["genome"] = chim["genome"].astype(str).map(tip_to_accession)

    top_list = chim.head(int(args.top_n))["genome"].tolist()
    top_set = set(top_list)

    g = per_genome_tail_metrics(ent)
    g["group"] = np.where(g["genome"].isin(top_set), "chimera_topN", "other")

    out_metrics = out_dir / "per_genome_entropy_tail_metrics.csv"
    g.to_csv(out_metrics, index=False)

    # Metrics to test (the story you asked for)
    metrics_to_test = [
        ("p95", "95th percentile entropy (bits)"),
        ("iqr", "Entropy IQR (p75 - p25)"),
        ("max_minus_median", "Max - Median entropy (bits)"),
        ("p95_minus_p05", "p95 - p05 entropy (bits)"),
        ("tail_excess_95", "p95 - median entropy (bits)"),
    ]

    summary_rows = []

    for idx, (mcol, label) in enumerate(metrics_to_test):
        top = g.loc[g["group"] == "chimera_topN", mcol].to_numpy(dtype=float)
        oth = g.loc[g["group"] == "other", mcol].to_numpy(dtype=float)

        diff, pval = permutation_test_two_sided(top, oth, stat=args.stat, perm=args.perm, seed=args.seed + idx)
        d_eff = cohen_d(top, oth)

        summary_rows.append(
            dict(
                metric=mcol,
                metric_label=label,
                top_n=int(args.top_n),
                perm=int(args.perm),
                seed=int(args.seed + idx),
                test_stat=args.stat,
                diff_top_minus_other=diff,
                p_two_sided=pval,
                cohen_d=d_eff,
                n_top=int(np.isfinite(top).sum()),
                n_other=int(np.isfinite(oth).sum()),
            )
        )

        note = [
            rf"{args.stat} diff (Top−Other) = {diff:.4g}",
            rf"$p$ = {pval:.4g}",
            rf"Cohen's d = {d_eff:.3f}" if np.isfinite(d_eff) else "Cohen's d = NA",
        ]

        boxplot_two_groups(
            out_pdf=plots_dir / f"tail_{mcol}_boxplot.pdf",
            out_png=plots_dir / f"tail_{mcol}_boxplot.png",
            others=oth,
            top=top,
            ylabel=label,
            title=f"Chimera Top{args.top_n} vs Others: {label}",
            note_lines=note,
            dpi=args.dpi,
        )

    summary = pd.DataFrame(summary_rows)
    out_summary = out_dir / "tail_permutation_tests_summary.csv"
    summary.to_csv(out_summary, index=False)

    print("Done.")
    print(f"- Per-genome metrics: {out_metrics}")
    print(f"- Test summary: {out_summary}")
    print(f"- Plots: {plots_dir}")


if __name__ == "__main__":
    main()
