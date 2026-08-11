"""
Tree/ORF distance versus mosaic-label agreement and mosaic-complexity scoring.

Inputs:
- ORF distance matrix: pairwise_identity_distance.csv
- Mosaic window clustering: window_assignments.csv

This script:
1) Summarizes mosaic labels per genome (dominant label, purity, entropy, switch count).
2) Clusters ORF distance matrix into K clusters (Agglomerative, precomputed distance).
3) Computes ARI / NMI between ORF clusters and mosaic dominant labels.
4) Builds contingency + ORF->mosaic label mapping (majority vote per ORF cluster).
5) Ranks genomes by a transparent mosaic-complexity score while retaining ORF–mosaic agreement diagnostics.

Example:
  python pipeline/cssv_tree_mosaic_agreement.py ^
    --orf_dist "results/orf3_phylogeny/pairwise_identity_distance.csv" ^
    --orf_name_split "|" ^
    --window_assignments "results/gb/window_assignments.csv" ^
    --out_dir "results/tree_mosaic_agreement" ^
    --k_orf 8 --min_purity 0.6 --top_n 20 ^
    --tree_newick "results/orf3_phylogeny/nj_tree.newick"
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def read_square_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda x: str(x).strip())
    df.columns = [str(c).strip() for c in df.columns]
    return df


def apply_split_labels(df: pd.DataFrame, split: Optional[str]) -> pd.DataFrame:
    if not split:
        return df
    idx = [str(x).split(split)[0] for x in df.index]
    cols = [str(x).split(split)[0] for x in df.columns]

    if len(set(idx)) != len(idx):
        dupes = sorted({x for x in idx if idx.count(x) > 1})
        raise ValueError(
            f"After splitting ORF labels by {split!r}, duplicates appeared: {dupes[:10]} "
            f"(showing up to 10). Use a matrix that has exactly 1 sequence per genome."
        )
    if len(set(cols)) != len(cols):
        dupes = sorted({x for x in cols if cols.count(x) > 1})
        raise ValueError(
            f"After splitting ORF column labels by {split!r}, duplicates appeared: {dupes[:10]} "
            f"(showing up to 10)."
        )

    out = df.copy()
    out.index = idx
    out.columns = cols
    if set(out.index) == set(out.columns):
        out = out.loc[out.index, out.index]
    return out


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Matrix is not square: {df.shape}")
    if list(df.index) != list(df.columns):
        if set(df.index) == set(df.columns):
            df = df.loc[df.index, df.index]
        else:
            raise ValueError("Row/column labels differ and are not the same set.")

    m = df.to_numpy(dtype=float)
    if not np.allclose(m, m.T, equal_nan=True):
        m2 = np.nanmean(np.stack([m, m.T]), axis=0)
        df = pd.DataFrame(m2, index=df.index, columns=df.columns)
    return df


def parse_window_assignments(path: Path) -> pd.DataFrame:
    """
    Supports:
      - name,start,end,label
      - genome,start,assigned_cluster
      - genome,start,label
    Returns standardized: genome,start,label
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    if "genome" in cols:
        genome_col = cols["genome"]
    elif "name" in cols:
        genome_col = cols["name"]
    else:
        raise ValueError(f"Cannot find genome/name column in {path}. Columns: {list(df.columns)}")

    if "start" in cols:
        start_col = cols["start"]
    else:
        raise ValueError(f"Cannot find start column in {path}. Columns: {list(df.columns)}")

    if "label" in cols:
        label_col = cols["label"]
    elif "assigned_cluster" in cols:
        label_col = cols["assigned_cluster"]
    elif "cluster" in cols:
        label_col = cols["cluster"]
    else:
        raise ValueError(f"Cannot find label/assigned_cluster column in {path}. Columns: {list(df.columns)}")

    out = df[[genome_col, start_col, label_col]].copy()
    out.columns = ["genome", "start", "label"]
    out["genome"] = out["genome"].astype(str).str.strip()
    out["label"] = out["label"].astype(int)
    out["start"] = pd.to_numeric(out["start"], errors="coerce").fillna(0).astype(int)
    return out


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def mosaic_summary_per_genome(win: pd.DataFrame, circular: bool = True) -> pd.DataFrame:
    """Summarize label composition and transitions for each genome.

    Circular tracks include the boundary between the last and first windows in
    both the switch count and the denominator. Linear tracks use n-1 adjacent
    boundaries.
    """
    rows = []
    for genome, g in win.groupby("genome"):
        g = g.sort_values("start")
        labels = g["label"].to_numpy(dtype=int)
        n = int(labels.size)
        if n == 0:
            continue

        uniq, cnt = np.unique(labels, return_counts=True)
        dominant_label = int(uniq[np.argmax(cnt)])
        dominant_frac = float(np.max(cnt) / n)
        entropy = shannon_entropy_from_counts(cnt)
        n_labels = int((cnt > 0).sum())

        if n > 1:
            switches = int(np.sum(labels[1:] != labels[:-1]))
            boundaries = n - 1
            if circular:
                switches += int(labels[-1] != labels[0])
                boundaries = n
            switch_rate = float(switches / boundaries) if boundaries > 0 else 0.0
        else:
            switches = 0
            boundaries = 0
            switch_rate = 0.0

        rows.append(
            {
                "genome": genome,
                "n_windows": n,
                "dominant_label": dominant_label,
                "dominant_frac": dominant_frac,
                "label_entropy_bits": entropy,
                "label_entropy": entropy,  # backward-compatible alias
                "n_labels_observed": n_labels,
                "n_labels": n_labels,      # backward-compatible alias
                "n_switches": switches,
                "n_adjacent_boundaries": boundaries,
                "switch_rate": switch_rate,
                "circular_barcode": bool(circular),
            }
        )
    return pd.DataFrame(rows).sort_values("genome").reset_index(drop=True)


def agglomerative_cluster_precomputed(dist: np.ndarray, n_clusters: int, linkage: str = "average") -> np.ndarray:
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage=linkage,
        )
    return model.fit_predict(dist)


def majority_mapping(orflab: np.ndarray, moslab: np.ndarray) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for c in sorted(set(orflab.tolist())):
        mask = orflab == c
        mos = moslab[mask]
        vals, cnt = np.unique(mos, return_counts=True)
        mapping[c] = int(vals[np.argmax(cnt)])
    return mapping


def safe_norm_entropy(entropy: float, n_labels_total: int) -> float:
    if n_labels_total <= 1:
        return 0.0
    return float(entropy / math.log2(n_labels_total))


def annotate_tree_newick(tree_path: Path, out_path: Path, genome_df: pd.DataFrame) -> None:
    try:
        from Bio import Phylo
    except Exception as e:
        raise RuntimeError("Biopython required (conda/pip install biopython).") from e

    tree = Phylo.read(str(tree_path), "newick")
    lookup = genome_df.set_index("genome").to_dict(orient="index")

    for term in tree.get_terminals():
        name = str(term.name).strip()
        key = name
        if key not in lookup and "|" in key:
            key = key.split("|")[0]
        if key in lookup:
            d = lookup[key]
            term.name = f"{key}|orfC{d['orf_cluster']}|mos{d['dominant_label']}|pur{d['dominant_frac']:.2f}"
        else:
            term.name = name

    Phylo.write(tree, str(out_path), "newick")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orf_dist", required=True, help="pairwise_identity_distance.csv (square matrix)")
    ap.add_argument("--window_assignments", required=True, help="window_assignments.csv from mosaic clustering")
    ap.add_argument("--out_dir", required=True, help="Output folder")

    ap.add_argument("--orf_name_split", default="|", help="Split ORF matrix labels by this and keep prefix. Use 'none' to disable.")
    ap.add_argument("--k_orf", type=int, default=None, help="ORF clusters K. Default: #unique mosaic dominant labels")
    ap.add_argument(
        "--mosaic_k", type=int, default=None,
        help="Total barcode-cluster K used to normalize entropy. Default: infer from all window labels.",
    )
    ap.add_argument(
        "--linear_barcode", action="store_true",
        help="Do not count the last-to-first barcode boundary. Circular is the default.",
    )
    ap.add_argument("--linkage", default="average", choices=["average", "complete", "single"])

    ap.add_argument("--scan_k_min", type=int, default=None)
    ap.add_argument("--scan_k_max", type=int, default=None)

    ap.add_argument("--min_purity", type=float, default=0.6)
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--tree_newick", default=None, help="Optional nj_tree.newick to annotate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    D = read_square_csv(Path(args.orf_dist))
    split = None if (args.orf_name_split in (None, "", "none", "None")) else args.orf_name_split
    D = apply_split_labels(D, split)
    D = symmetrize(D)

    win = parse_window_assignments(Path(args.window_assignments))
    circular_barcode = not args.linear_barcode
    mos = mosaic_summary_per_genome(win, circular=circular_barcode)

    common = sorted(set(D.index) & set(mos["genome"]))
    missing_in_mosaic = sorted(set(D.index) - set(mos["genome"]))
    missing_in_orf = sorted(set(mos["genome"]) - set(D.index))

    if len(common) < 3:
        raise ValueError(
            f"Too few overlapping genomes (common={len(common)}). "
            f"Missing in mosaic(from ORF): {missing_in_mosaic[:10]} ; "
            f"Missing in ORF(from mosaic): {missing_in_orf[:10]}"
        )

    D = D.loc[common, common]
    mos = mos[mos["genome"].isin(common)].copy().set_index("genome").loc[common].reset_index()

    dist = D.to_numpy(dtype=float)
    if not np.isfinite(dist).all():
        good = np.isfinite(dist).all(axis=1)
        kept = [g for g, ok in zip(common, good.tolist()) if ok]
        dropped = [g for g, ok in zip(common, good.tolist()) if not ok]
        if len(kept) < 3:
            raise ValueError("After dropping NaN/inf genomes, too few remain.")
        dist = dist[np.ix_(good, good)]
        mos = mos[mos["genome"].isin(kept)].copy()
        common = kept
        missing_in_mosaic += dropped
        D = D.loc[common, common]

    if args.k_orf is None:
        k = int(mos["dominant_label"].nunique())
        if k < 2:
            k = 2
    else:
        k = int(args.k_orf)

    scan_rows = []
    if args.scan_k_min is not None and args.scan_k_max is not None:
        for kk in range(int(args.scan_k_min), int(args.scan_k_max) + 1):
            if kk < 2 or kk > len(common):
                continue
            orf_lab = agglomerative_cluster_precomputed(dist, n_clusters=kk, linkage=args.linkage)
            ari = float(adjusted_rand_score(mos["dominant_label"], orf_lab))
            nmi = float(normalized_mutual_info_score(mos["dominant_label"], orf_lab))
            scan_rows.append({"k": kk, "ARI": ari, "NMI": nmi})
        pd.DataFrame(scan_rows).to_csv(out_dir / "agreement_metrics.k_scan.csv", index=False)

    orf_cluster = agglomerative_cluster_precomputed(dist, n_clusters=k, linkage=args.linkage)
    moslab = mos["dominant_label"].to_numpy(dtype=int)

    ari = float(adjusted_rand_score(moslab, orf_cluster))
    nmi = float(normalized_mutual_info_score(moslab, orf_cluster))

    metrics = pd.DataFrame(
        [
            {
                "n_genomes": len(common),
                "k_orf": k,
                "k_mosaic_dominant_labels": int(mos["dominant_label"].nunique()),
                "circular_barcode": bool(circular_barcode),
                "ARI": ari,
                "NMI": nmi,
                "linkage": args.linkage,
                "missing_in_mosaic_count": len(missing_in_mosaic),
                "missing_in_orf_count": len(missing_in_orf),
            }
        ]
    )
    metrics.to_csv(out_dir / "agreement_metrics.csv", index=False)
    mos.to_csv(out_dir / "mosaic_per_genome.csv", index=False)

    orf_df = pd.DataFrame({"genome": common, "orf_cluster": orf_cluster}).sort_values("genome")
    orf_df.to_csv(out_dir / "orf_clusters_per_genome.csv", index=False)

    merged = mos.merge(orf_df, on="genome", how="left")
    contingency = pd.crosstab(merged["orf_cluster"], merged["dominant_label"])
    contingency.to_csv(out_dir / "contingency_orf_vs_mosaic.csv")

    mapping = majority_mapping(merged["orf_cluster"].to_numpy(int), merged["dominant_label"].to_numpy(int))
    merged["mapped_mosaic_from_orf"] = merged["orf_cluster"].map(mapping).astype(int)
    merged["mismatch"] = merged["mapped_mosaic_from_orf"] != merged["dominant_label"]

    observed_window_labels = int(win["label"].nunique())
    n_labels_total = int(args.mosaic_k) if args.mosaic_k is not None else observed_window_labels
    if n_labels_total < observed_window_labels:
        raise ValueError(
            f"--mosaic_k={n_labels_total} is smaller than the {observed_window_labels} labels "
            "observed in window_assignments.csv"
        )
    entropy_max_bits = math.log2(n_labels_total) if n_labels_total > 1 else 0.0
    merged["entropy_log_base"] = 2
    merged["entropy_normalization_k"] = n_labels_total
    merged["entropy_max_bits"] = entropy_max_bits
    merged["norm_entropy"] = merged["label_entropy_bits"].apply(
        lambda e: safe_norm_entropy(float(e), n_labels_total)
    )
    if (merged["norm_entropy"] > 1.0 + 1e-10).any():
        raise ValueError("Normalized entropy exceeded 1; verify mosaic K and label assignments.")

    # Composite mosaic-complexity score.
    # ORF–mosaic mismatch is retained as a separate diagnostic and is not
    # included in the ranking score. The score therefore summarizes only
    # barcode-derived properties: low purity, normalized entropy, and
    # circular switch rate.
    merged["mosaic_complexity_score"] = (
        (1.0 - merged["dominant_frac"])
        + merged["norm_entropy"]
        + merged["switch_rate"]
    )
    merged["flag_low_purity"] = merged["dominant_frac"] < float(args.min_purity)

    ranked = merged.sort_values(
        ["mosaic_complexity_score", "dominant_frac", "switch_rate", "genome"],
        ascending=[False, True, False, True],
    )
    ranked.to_csv(out_dir / "mosaic_orf_merged_per_genome.csv", index=False)
    ranked.head(int(args.top_n)).to_csv(out_dir / "mosaic_ranked_genomes.csv", index=False)

    if missing_in_mosaic:
        (out_dir / "missing_in_mosaic.txt").write_text("\n".join(missing_in_mosaic) + "\n", encoding="utf-8")
    if missing_in_orf:
        (out_dir / "missing_in_orf.txt").write_text("\n".join(missing_in_orf) + "\n", encoding="utf-8")

    if args.tree_newick:
        annotate_tree_newick(Path(args.tree_newick), out_dir / "annotated_tree.newick", merged)

    print("Done.")
    print(f"- Metrics:      {out_dir / 'agreement_metrics.csv'}")
    print(f"- Mosaic:       {out_dir / 'mosaic_per_genome.csv'}")
    print(f"- ORF clusters: {out_dir / 'orf_clusters_per_genome.csv'}")
    print(f"- Contingency:  {out_dir / 'contingency_orf_vs_mosaic.csv'}")
    print(f"- Highest-scoring genomes: {out_dir / 'mosaic_ranked_genomes.csv'}")
    print(f"- Full table:   {out_dir / 'mosaic_orf_merged_per_genome.csv'}")
    if args.tree_newick:
        print(f"- Tree:         {out_dir / 'annotated_tree.newick'}")


if __name__ == "__main__":
    main()
