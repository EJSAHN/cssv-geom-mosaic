"""
Re-plot distance correlation scatter as vector PDF + PNG 300dpi.

Inputs:
- --matrix_a: distance matrix A (CSV square)
- --matrix_b: distance matrix B (CSV square)
- --summary: summary CSV produced by cssv_compare_distances.py (contains r,p,n_pairs etc.)
- --out_prefix: output prefix (no extension)
Options:
- --b_split: split matrix B labels at this string (e.g. '|') to map to genome accessions.

Outputs:
- <out_prefix>.pdf  (vector, editable text)
- <out_prefix>.png  (300 dpi)

Usage:
python scripts/cssv_plot_distance_correlation_vector.py ^
  --matrix_a "results/gb/k4_cosine_distance.csv" ^
  --matrix_b "results/orf3_phylogeny/pairwise_identity_distance.csv" ^
  --summary  "results/compare_distances/compare_distances.summary.csv" ^
  --b_split "|" ^
  --out_prefix "results/compare_distances/compare_distances.scatter_vector"
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def read_sq(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda x: str(x).strip())
    df.columns = [str(c).strip() for c in df.columns]
    return df


def apply_split(df: pd.DataFrame, split: str | None) -> pd.DataFrame:
    if not split:
        return df
    idx = [str(x).split(split)[0] for x in df.index]
    cols = [str(x).split(split)[0] for x in df.columns]
    if len(set(idx)) != len(idx) or len(set(cols)) != len(cols):
        raise ValueError("Label split created duplicates; use a matrix with 1 sequence per genome.")
    out = df.copy()
    out.index = idx
    out.columns = cols
    return out


def upper_vec(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_a", required=True)
    ap.add_argument("--matrix_b", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--b_split", default=None)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--point_size", type=float, default=16.0)
    args = ap.parse_args()

    A = read_sq(Path(args.matrix_a))
    B = read_sq(Path(args.matrix_b))
    B = apply_split(B, args.b_split)

    common = sorted(set(A.index) & set(B.index))
    if len(common) < 3:
        raise ValueError(f"Too few common labels (n={len(common)})")

    A = A.loc[common, common]
    B = B.loc[common, common]

    a = A.to_numpy(dtype=float)
    b = B.to_numpy(dtype=float)
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(b, 0.0)

    va = upper_vec(a)
    vb = upper_vec(b)

    summ = pd.read_csv(Path(args.summary))
    row = summ.iloc[0].to_dict()

    r = row.get("r", None)
    p = row.get("p_two_sided", row.get("p", None))
    n_pairs = row.get("n_pairs", len(va))

    title = f"spearman r={float(r):.3f}, p={float(p):.4g}, n={int(n_pairs)} pairs" if r is not None else "distance correlation"

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111)
    ax.scatter(va, vb, s=args.point_size, alpha=args.alpha)
    ax.set_xlabel("Matrix A (upper triangle)")
    ax.set_ylabel("Matrix B (upper triangle)")
    ax.set_title(title)
    fig.tight_layout()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Done.")
    print(f"- {out_prefix.with_suffix('.pdf')}")
    print(f"- {out_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
