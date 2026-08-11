#!/usr/bin/env python3
"""Compute reproducible diagnostics for a two-dimensional MDS embedding."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance_matrix", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.distance_matrix, index_col=0)
    d.index = d.index.astype(str)
    d.columns = d.columns.astype(str)
    e = pd.read_csv(args.embedding)
    name_col = "name" if "name" in e.columns else e.columns[0]
    coord_cols = [c for c in e.columns if c.lower() in {"mds1", "mds2"}]
    if len(coord_cols) != 2:
        raise ValueError(f"Expected mds1/mds2 columns; got {list(e.columns)}")
    e[name_col] = e[name_col].astype(str)
    e = e.set_index(name_col)
    common = [x for x in d.index if x in e.index]
    if len(common) < 3:
        raise ValueError("Too few common labels between distance matrix and embedding")
    d = d.loc[common, common].apply(pd.to_numeric, errors="coerce")
    coords = e.loc[common, coord_cols].to_numpy(float)
    original = squareform(d.to_numpy(float), checks=False)
    embedded = pdist(coords, metric="euclidean")
    residual = original - embedded
    raw_stress = float(np.sum(residual ** 2))
    stress1 = float(np.sqrt(raw_stress / np.sum(original ** 2)))
    p_r = float(pearsonr(original, embedded).statistic)
    s_r = float(spearmanr(original, embedded).statistic)
    summary = pd.DataFrame([{
        "n_genomes": len(common),
        "n_pairs": len(original),
        "n_dimensions": 2,
        "raw_stress": raw_stress,
        "kruskal_stress_1": stress1,
        "pearson_original_vs_embedded_distance": p_r,
        "spearman_original_vs_embedded_distance": s_r,
        "axis_interpretation": "arbitrary orthogonal MDS coordinates; not percentages of explained variance",
    }])
    summary.to_csv(out / "mds_embedding_diagnostics.csv", index=False)
    pair_df = pd.DataFrame({
        "original_distance": original,
        "embedded_2d_distance": embedded,
        "residual": residual,
    })
    pair_df.to_csv(out / "mds_distance_reconstruction.csv", index=False)
    print(f"[OK] Kruskal Stress-1={stress1:.6f}; Pearson r={p_r:.6f}")


if __name__ == "__main__":
    main()
