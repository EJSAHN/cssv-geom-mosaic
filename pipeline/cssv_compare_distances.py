"""
Compare two genome/ORF distance matrices (e.g., k-mer cosine vs ORF pairwise identity distance)
and run a Mantel-style permutation test.

Typical use:
  python pipeline/cssv_compare_distances.py ^
    --matrix_a "results/gb/k4_cosine_distance.csv" ^
    --matrix_b "results/orf3_phylogeny/pairwise_identity_distance.csv" ^
    --b_split "|" ^
    --out_dir  "results/compare_distances" ^
    --method spearman --perm 5000 --seed 0 --plot

Outputs:
  - compare_distances.summary.csv
  - compare_distances.genomes_used.txt
  - (optional) compare_distances.scatter.png  (use --plot)

Notes:
- For ORF matrices where labels look like "MN179342.1|top1|...", use --b_split "|"
  so we compare on the genome accession prefix.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _read_square_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda x: str(x).strip())
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _apply_split(df: pd.DataFrame, split: Optional[str]) -> pd.DataFrame:
    if not split:
        return df
    new_index = [str(x).split(split)[0] for x in df.index]
    new_cols = [str(x).split(split)[0] for x in df.columns]

    if len(set(new_index)) != len(new_index):
        dupes = sorted({x for x in new_index if new_index.count(x) > 1})
        raise ValueError(
            f"After splitting index by {split!r}, duplicate labels appeared: {dupes[:10]} "
            f"(showing up to 10). This usually means you have >1 sequence per genome."
        )
    if len(set(new_cols)) != len(new_cols):
        dupes = sorted({x for x in new_cols if new_cols.count(x) > 1})
        raise ValueError(
            f"After splitting columns by {split!r}, duplicate labels appeared: {dupes[:10]} "
            f"(showing up to 10)."
        )

    out = df.copy()
    out.index = new_index
    out.columns = new_cols
    return out


def _symmetrize(df: pd.DataFrame, how: str = "average") -> pd.DataFrame:
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Matrix is not square: {df.shape}")
    if list(df.index) != list(df.columns):
        if set(df.index) == set(df.columns):
            df = df.loc[df.index, df.index]
        else:
            raise ValueError("Row/column labels differ and are not the same set.")

    m = df.to_numpy(dtype=float)
    if not np.allclose(m, m.T, equal_nan=True):
        if how == "average":
            m2 = np.nanmean(np.stack([m, m.T]), axis=0)
        elif how == "lower":
            m2 = np.tril(m) + np.tril(m, -1).T
        elif how == "upper":
            m2 = np.triu(m) + np.triu(m, 1).T
        else:
            raise ValueError(f"Unknown symmetrize mode: {how}")
        df = pd.DataFrame(m2, index=df.index, columns=df.columns)
    return df


def _drop_incomplete(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    A = a.to_numpy(dtype=float)
    B = b.to_numpy(dtype=float)
    good = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    keep = a.index[good].tolist()
    return a.loc[keep, keep], b.loc[keep, keep]


def _vectorize_upper(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(rx, ry)


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    method = method.lower()
    if method == "pearson":
        return _pearson(x, y)
    if method == "spearman":
        return _spearman(x, y)
    raise ValueError(f"Unknown method: {method} (choose pearson/spearman)")


def mantel_test(
    a: np.ndarray,
    b: np.ndarray,
    method: str = "spearman",
    perm: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    if b.shape != (n, n):
        raise ValueError("A and B must have same shape")

    va = _vectorize_upper(a)
    vb = _vectorize_upper(b)
    r_obs = _corr(va, vb, method=method)

    r_perm = np.empty(perm, dtype=float)
    for i in range(perm):
        pidx = rng.permutation(n)
        bp = b[np.ix_(pidx, pidx)]
        r_perm[i] = _corr(va, _vectorize_upper(bp), method=method)

    p = (1.0 + float(np.sum(np.abs(r_perm) >= abs(r_obs)))) / (perm + 1.0)
    z = float((r_obs - np.mean(r_perm)) / (np.std(r_perm, ddof=1) + 1e-12))
    return r_obs, p, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_a", required=True, help="CSV square matrix A (index_col=0)")
    ap.add_argument("--matrix_b", required=True, help="CSV square matrix B (index_col=0)")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--method", choices=["pearson", "spearman"], default="spearman")
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--a_split", default=None, help="Split A labels at this string and keep prefix")
    ap.add_argument("--b_split", default=None, help="Split B labels at this string and keep prefix (e.g. '|')")
    ap.add_argument("--symmetrize", choices=["average", "lower", "upper"], default="average")
    ap.add_argument("--plot", action="store_true", help="Also write a scatter plot PNG")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A = _read_square_csv(Path(args.matrix_a))
    B = _read_square_csv(Path(args.matrix_b))

    A = _apply_split(A, args.a_split)
    B = _apply_split(B, args.b_split)

    A = _symmetrize(A, how=args.symmetrize)
    B = _symmetrize(B, how=args.symmetrize)

    common = sorted(set(A.index) & set(B.index))
    if len(common) < 3:
        raise ValueError(
            f"Too few common labels between matrices (common={len(common)}). "
            f"Example A labels: {list(A.index)[:5]}, B labels: {list(B.index)[:5]}"
        )

    A = A.loc[common, common]
    B = B.loc[common, common]

    A, B = _drop_incomplete(A, B)
    names_used = A.index.tolist()

    a = A.to_numpy(dtype=float)
    b = B.to_numpy(dtype=float)

    r, p, z = mantel_test(a, b, method=args.method, perm=args.perm, seed=args.seed)

    summary = pd.DataFrame(
        [
            {
                "method": args.method,
                "r": r,
                "p_two_sided": p,
                "z": z,
                "perm": args.perm,
                "seed": args.seed,
                "n_genomes_used": len(names_used),
                "n_pairs": int(len(names_used) * (len(names_used) - 1) / 2),
            }
        ]
    )
    summary_path = out_dir / "compare_distances.summary.csv"
    summary.to_csv(summary_path, index=False)

    names_path = out_dir / "compare_distances.genomes_used.txt"
    names_path.write_text("\n".join(names_used) + "\n", encoding="utf-8")

    print("Done.")
    print(f"- Summary: {summary_path}")
    print(f"- Genomes used: {names_path}")

    if args.plot:
        import matplotlib.pyplot as plt

        va = _vectorize_upper(a)
        vb = _vectorize_upper(b)
        fig = plt.figure()
        plt.scatter(va, vb, s=12)
        plt.xlabel("Matrix A (upper triangle)")
        plt.ylabel("Matrix B (upper triangle)")
        plt.title(f"{args.method} r={r:.3f}, p={p:.3g}, n={len(va)} pairs")
        plot_path = out_dir / "compare_distances.scatter.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"- Plot: {plot_path}")


if __name__ == "__main__":
    main()
