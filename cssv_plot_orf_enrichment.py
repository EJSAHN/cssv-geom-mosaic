"""
Plot ORF enrichment outputs produced by cssv_mosaic_orf_analysis.py

Inputs (from your mosaic_orf output folder):
- enrichment_summary.csv
- null_fracs.npy
- switchpoint_orf_distances.csv

Outputs (to --out_dir):
- null_frac_hist.(pdf/png): null distribution of "fraction of switchpoints near ORF boundary"
- dist_hist.(pdf/png): histogram of distances to nearest ORF boundary (bp)
- dist_cdf.(pdf/png): CDF of distances
- per_genome_median_dist.(pdf/png): per-genome median distance (top N genomes)

No seaborn. Matplotlib only. Headless-safe (Agg backend).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # important for CLI / servers / no-display
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Pick first existing column from candidates; else raise."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns exist. candidates={candidates}, cols={list(df.columns)}")


def plot_null_distribution(
    summary_df: pd.DataFrame,
    null_fracs: np.ndarray,
    out_dir: Path,
    prefix: str,
    bins: int,
    fmt: str,
    dpi: int,
) -> None:
    # Summary may be 1-row or multi-row; use first row
    row = summary_df.iloc[0].to_dict()

    observed = float(row.get("observed_frac_near", np.nan))
    p_emp = row.get("p_empirical", None)
    z = row.get("z", None)
    near_bp = row.get("near_bp", None)
    n_perm = row.get("n_perm", len(null_fracs))
    n_switch = row.get("n_switchpoints", None)

    # If p_empirical missing, compute one-sided enrichment p = P(null >= obs)
    null_clean = np.asarray(null_fracs, dtype=float)
    null_clean = null_clean[np.isfinite(null_clean)]
    if p_emp is None or (isinstance(p_emp, float) and np.isnan(p_emp)):
        if np.isfinite(observed) and len(null_clean) > 0:
            p_emp = (1.0 + float(np.sum(null_clean >= observed))) / (len(null_clean) + 1.0)
        else:
            p_emp = np.nan

    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)

    ax.hist(null_clean, bins=bins)
    if np.isfinite(observed):
        ax.axvline(observed, linewidth=2)

    title_parts = ["Null distribution: fraction of switchpoints near ORF boundary"]
    if near_bp is not None and str(near_bp) != "nan":
        title_parts.append(f"(near_bp={near_bp})")
    ax.set_title(" ".join(title_parts))
    ax.set_xlabel("null frac (per permutation)")
    ax.set_ylabel("count")

    # Annotate
    text_lines = []
    if np.isfinite(observed):
        text_lines.append(f"observed={observed:.4f}")
    if np.isfinite(float(p_emp)):
        text_lines.append(f"p_empirical={float(p_emp):.4g}")
    if z is not None and str(z) != "nan":
        try:
            text_lines.append(f"z={float(z):.3f}")
        except Exception:
            pass
    if n_switch is not None and str(n_switch) != "nan":
        try:
            text_lines.append(f"n_switchpoints={int(n_switch)}")
        except Exception:
            pass
    if n_perm is not None and str(n_perm) != "nan":
        try:
            text_lines.append(f"n_perm={int(n_perm)}")
        except Exception:
            pass

    if text_lines:
        ax.text(
            0.98,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.2),
        )

    fig.tight_layout()
    out_base = out_dir / f"{prefix}_null_frac_hist"
    if fmt in ("pdf", "both"):
        fig.savefig(str(out_base) + ".pdf")
    if fmt in ("png", "both"):
        fig.savefig(str(out_base) + ".png", dpi=dpi)
    plt.close(fig)


def plot_distance_hist_and_cdf(
    dist_df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    bins: int,
    fmt: str,
    dpi: int,
) -> None:
    dist_col = _pick_col(
        dist_df,
        candidates=[
            "dist_to_nearest_orf_boundary_bp",
            "distance_bp",
            "dist_bp",
            "dist",
        ],
    )

    d = pd.to_numeric(dist_df[dist_col], errors="coerce").dropna().values.astype(float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        raise ValueError(f"No finite values found in distance column '{dist_col}'")

    # Histogram
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    ax.hist(d, bins=bins)
    ax.set_title("Distances to nearest ORF boundary (bp)")
    ax.set_xlabel("distance (bp)")
    ax.set_ylabel("count")
    fig.tight_layout()

    out_base = out_dir / f"{prefix}_dist_hist"
    if fmt in ("pdf", "both"):
        fig.savefig(str(out_base) + ".pdf")
    if fmt in ("png", "both"):
        fig.savefig(str(out_base) + ".png", dpi=dpi)
    plt.close(fig)

    # CDF
    d_sorted = np.sort(d)
    y = np.arange(1, len(d_sorted) + 1) / float(len(d_sorted))

    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(d_sorted, y)
    ax.set_title("CDF: distance to nearest ORF boundary")
    ax.set_xlabel("distance (bp)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    out_base = out_dir / f"{prefix}_dist_cdf"
    if fmt in ("pdf", "both"):
        fig.savefig(str(out_base) + ".pdf")
    if fmt in ("png", "both"):
        fig.savefig(str(out_base) + ".png", dpi=dpi)
    plt.close(fig)


def plot_per_genome_median_distance(
    dist_df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    top_n: int,
    fmt: str,
    dpi: int,
) -> None:
    name_col = _pick_col(dist_df, candidates=["name", "genome", "genome_name"])
    dist_col = _pick_col(
        dist_df,
        candidates=[
            "dist_to_nearest_orf_boundary_bp",
            "distance_bp",
            "dist_bp",
            "dist",
        ],
    )

    tmp = dist_df[[name_col, dist_col]].copy()
    tmp[dist_col] = pd.to_numeric(tmp[dist_col], errors="coerce")
    tmp = tmp.dropna()

    if tmp.empty:
        return

    med = (
        tmp.groupby(name_col, sort=False)[dist_col]
        .median()
        .sort_values(ascending=True)
    )

    # If too many genomes, show top_n (closest medians or just first N after sorting)
    if top_n is not None and top_n > 0 and len(med) > top_n:
        med_plot = med.iloc[:top_n]
    else:
        med_plot = med

    fig = plt.figure(figsize=(10, 5.2))
    ax = fig.add_subplot(111)
    ax.bar(range(len(med_plot)), med_plot.values)
    ax.set_title(f"Per-genome median distance to ORF boundary (top {len(med_plot)})")
    ax.set_ylabel("median distance (bp)")
    ax.set_xticks(range(len(med_plot)))
    ax.set_xticklabels(med_plot.index.tolist(), rotation=90)
    fig.tight_layout()

    out_base = out_dir / f"{prefix}_per_genome_median_dist"
    if fmt in ("pdf", "both"):
        fig.savefig(str(out_base) + ".pdf")
    if fmt in ("png", "both"):
        fig.savefig(str(out_base) + ".png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrichment_summary", required=True, help="Path to enrichment_summary.csv")
    ap.add_argument("--null_fracs", required=True, help="Path to null_fracs.npy")
    ap.add_argument("--distances", required=True, help="Path to switchpoint_orf_distances.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder for plots")
    ap.add_argument("--prefix", default="orf_enrichment", help="Filename prefix for outputs")
    ap.add_argument("--bins", type=int, default=50, help="Histogram bins")
    ap.add_argument("--top_genomes", type=int, default=20, help="Top N genomes to show in per-genome plot (sorted by median distance)")
    ap.add_argument("--fmt", choices=["pdf", "png", "both"], default="pdf", help="Output format")
    ap.add_argument("--dpi", type=int, default=200, help="DPI for PNG outputs")
    ap.add_argument("--skip_per_genome", action="store_true", help="Skip per-genome median plot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    summary_df = pd.read_csv(args.enrichment_summary)
    null_fracs = np.load(args.null_fracs, allow_pickle=False)
    dist_df = pd.read_csv(args.distances)

    plot_null_distribution(
        summary_df=summary_df,
        null_fracs=null_fracs,
        out_dir=out_dir,
        prefix=args.prefix,
        bins=args.bins,
        fmt=args.fmt,
        dpi=args.dpi,
    )

    plot_distance_hist_and_cdf(
        dist_df=dist_df,
        out_dir=out_dir,
        prefix=args.prefix,
        bins=args.bins,
        fmt=args.fmt,
        dpi=args.dpi,
    )

    if not args.skip_per_genome:
        plot_per_genome_median_distance(
            dist_df=dist_df,
            out_dir=out_dir,
            prefix=args.prefix,
            top_n=args.top_genomes,
            fmt=args.fmt,
            dpi=args.dpi,
        )

    print("Done.")
    print(f"- plots saved in: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
