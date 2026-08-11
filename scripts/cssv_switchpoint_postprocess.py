"""
Post-process mosaic switchpoints across genomes.

This script is tolerant to different switchpoint CSV schemas and normalizes to:
  name, pos0, from_label, to_label, run_len

Accepted switchpoint schemas include:
A) name, pos0, from_label, to_label, run_len
B) name, pos0, prev_label, new_label, prev_run_windows, new_run_windows
   -> from_label=prev_label, to_label=new_label
   -> run_len = min(prev_run_windows, new_run_windows)

Inputs
- switchpoints.csv : from your mosaic pipeline / cssv_mosaic_orf_analysis.py
- genome_summary.csv : from cssv_gb_pipeline.py (recommended; provides genome length)

Outputs (under --out_dir)
- switchpoints_per_genome.csv
- switchpoint_transition_counts.csv
- switchpoint_transition_matrix.csv
- switchpoint_position_density.csv
- (optional) plots/*.pdf

Usage:
python scripts/cssv_switchpoint_postprocess.py ^
  --switchpoints "results/mosaic_orf/switchpoints.csv" ^
  --genome_summary "results/gb/genome_summary.csv" ^
  --out_dir "results/switchpoint_post" ^
  --bins 120 --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def normalize_switchpoints(sw: pd.DataFrame) -> pd.DataFrame:
    sw = sw.copy()

    # Schema B: prev_label/new_label
    if {"prev_label", "new_label"}.issubset(sw.columns) and not {"from_label", "to_label"}.issubset(sw.columns):
        sw = sw.rename(columns={"prev_label": "from_label", "new_label": "to_label"})

    # run length
    if "run_len" not in sw.columns:
        if {"prev_run_windows", "new_run_windows"}.issubset(sw.columns):
            sw["run_len"] = np.minimum(sw["prev_run_windows"].astype(float), sw["new_run_windows"].astype(float))
        elif "run_windows" in sw.columns:
            sw["run_len"] = sw["run_windows"].astype(float)

    required = {"name", "pos0", "from_label", "to_label", "run_len"}
    missing = sorted(list(required - set(sw.columns)))
    if missing:
        raise ValueError(
            f"switchpoints.csv missing required columns after normalization: {missing}. "
            f"Found columns: {list(sw.columns)}"
        )

    sw["pos0"] = sw["pos0"].astype(int)
    return sw


def load_inputs(switchpoints_csv: Path, genome_summary_csv: Optional[Path]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    sw = pd.read_csv(switchpoints_csv)
    sw = normalize_switchpoints(sw)

    gs = None
    if genome_summary_csv is not None:
        gs = pd.read_csv(genome_summary_csv)
        if "name" not in gs.columns or "length" not in gs.columns:
            raise ValueError("genome_summary.csv must contain columns: name, length")
    return sw, gs


def add_genome_lengths(sw: pd.DataFrame, gs: Optional[pd.DataFrame]) -> pd.DataFrame:
    sw = sw.copy()
    if "genome_len" in sw.columns:
        sw["genome_len"] = sw["genome_len"].astype(int)
        return sw

    if gs is None:
        raise ValueError("genome_len not present in switchpoints.csv, so you must pass --genome_summary")

    lens = gs[["name", "length"]].rename(columns={"length": "genome_len"})
    sw = sw.merge(lens, on="name", how="left")

    missing = sw["genome_len"].isna()
    if missing.any():
        missing_names = sorted(sw.loc[missing, "name"].unique().tolist())
        raise ValueError(
            "Some genomes in switchpoints.csv were not found in genome_summary.csv: "
            + ", ".join(missing_names[:20])
            + (" ..." if len(missing_names) > 20 else "")
        )

    sw["genome_len"] = sw["genome_len"].astype(int)
    return sw


def per_genome_summary(sw: pd.DataFrame) -> pd.DataFrame:
    trans_per_genome = (
        sw.assign(_pair=sw["from_label"].astype(str) + "->" + sw["to_label"].astype(str))
        .groupby("name")["_pair"]
        .nunique()
        .rename("n_unique_transitions")
    )

    agg = sw.groupby("name").agg(
        n_switchpoints=("pos0", "size"),
        mean_run=("run_len", "mean"),
        median_run=("run_len", "median"),
        min_run=("run_len", "min"),
        max_run=("run_len", "max"),
        genome_len=("genome_len", "first"),
    )
    out = agg.join(trans_per_genome, how="left").reset_index()
    out["switchpoints_per_kb"] = out["n_switchpoints"] / (out["genome_len"] / 1000.0)
    return out.sort_values(["n_switchpoints", "name"], ascending=[False, True]).reset_index(drop=True)


def transition_counts(sw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        sw.groupby(["from_label", "to_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    matrix = counts.pivot_table(index="from_label", columns="to_label", values="count", fill_value=0)
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    matrix_df = matrix.reset_index().rename_axis(None, axis=1)
    return counts, matrix_df


def position_density(sw: pd.DataFrame, bins: int) -> pd.DataFrame:
    if bins < 5 or bins > 10000:
        raise ValueError("--bins should be between 5 and 10000")

    pos_norm = sw["pos0"].astype(float) / sw["genome_len"].astype(float)
    pos_norm = pos_norm.clip(lower=0.0, upper=np.nextafter(1.0, 0.0))

    hist, edges = np.histogram(pos_norm.values, bins=bins, range=(0.0, 1.0))
    total = hist.sum()

    df = pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "count": hist,
            "density": (hist / total) if total else np.nan,
        }
    )
    df["bin_mid"] = (df["bin_start"] + df["bin_end"]) / 2.0
    return df


def maybe_plot(dens_df: pd.DataFrame, trans_matrix: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dens_df["bin_mid"], dens_df["density"])
    ax.set_xlabel("Normalized genome position (pos0 / genome_len)")
    ax.set_ylabel("Density")
    ax.set_title("Switchpoint density across genomes")
    fig.tight_layout()
    fig.savefig(plot_dir / "switchpoint_position_density.pdf")
    plt.close(fig)

    mat = trans_matrix.set_index("from_label")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([str(c) for c in mat.columns], rotation=90)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([str(i) for i in mat.index])
    ax.set_xlabel("to_label")
    ax.set_ylabel("from_label")
    ax.set_title("Switchpoint transition counts")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "switchpoint_transition_matrix.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--switchpoints", required=True, help="Path to switchpoints.csv")
    ap.add_argument("--genome_summary", default=None, help="Path to genome_summary.csv (recommended)")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins for normalized position density")
    ap.add_argument("--plot", action="store_true", help="Also write PDF plots (matplotlib)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sw, gs = load_inputs(Path(args.switchpoints), Path(args.genome_summary) if args.genome_summary else None)
    sw = add_genome_lengths(sw, gs)

    per_g = per_genome_summary(sw)
    trans_counts, trans_matrix = transition_counts(sw)
    dens = position_density(sw, bins=args.bins)

    per_g.to_csv(out_dir / "switchpoints_per_genome.csv", index=False)
    trans_counts.to_csv(out_dir / "switchpoint_transition_counts.csv", index=False)
    trans_matrix.to_csv(out_dir / "switchpoint_transition_matrix.csv", index=False)
    dens.to_csv(out_dir / "switchpoint_position_density.csv", index=False)

    if args.plot:
        maybe_plot(dens, trans_matrix, out_dir)

    print("Done.")
    print(f"- {out_dir / 'switchpoints_per_genome.csv'}")
    print(f"- {out_dir / 'switchpoint_transition_counts.csv'}")
    print(f"- {out_dir / 'switchpoint_transition_matrix.csv'}")
    print(f"- {out_dir / 'switchpoint_position_density.csv'}")
    if args.plot:
        print(f"- {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
