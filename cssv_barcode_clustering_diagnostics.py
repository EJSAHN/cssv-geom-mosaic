#!/usr/bin/env python3
"""Data-driven diagnostics for sliding-window barcode clustering.

Evaluates K using compactness and stability metrics and compares the baseline
Euclidean KMeans partition with spherical (cosine) k-means on L2-normalized
k-mer-frequency vectors. Outputs are tabular only.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


def parse_ints(text: str) -> List[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        a, b = [int(x) for x in text.split("-", 1)]
        return list(range(a, b + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def spherical_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int = 5,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, float, int]:
    Xn = normalize(X, norm="l2", axis=1, copy=True)
    best_labels = None
    best_loss = np.inf
    best_iter = 0
    rng = np.random.default_rng(random_state)
    for init_no in range(n_init):
        seed = int(rng.integers(0, 2**31 - 1))
        centers, _ = kmeans_plusplus(Xn, n_clusters=n_clusters, random_state=seed)
        centers = normalize(centers, norm="l2", axis=1, copy=False)
        previous_loss = np.inf
        labels = np.zeros(len(Xn), dtype=int)
        for iteration in range(1, max_iter + 1):
            similarities = Xn @ centers.T
            labels_new = np.argmax(similarities, axis=1)
            max_similarity = similarities[np.arange(len(Xn)), labels_new]
            loss = float(np.sum(1.0 - max_similarity))
            new_centers = np.zeros_like(centers)
            for j in range(n_clusters):
                members = Xn[labels_new == j]
                if len(members) == 0:
                    # Re-seed an empty cluster with a poorly represented window.
                    idx = int(np.argmin(max_similarity))
                    new_centers[j] = Xn[idx]
                else:
                    new_centers[j] = members.mean(axis=0)
            new_centers = normalize(new_centers, norm="l2", axis=1, copy=False)
            converged = np.array_equal(labels_new, labels) or abs(previous_loss - loss) <= tol
            labels = labels_new
            centers = new_centers
            previous_loss = loss
            if converged:
                break
        if previous_loss < best_loss:
            best_loss = previous_loss
            best_labels = labels.copy()
            best_iter = iteration
    if best_labels is None:
        raise RuntimeError("Spherical k-means failed")
    return best_labels, best_loss, best_iter


def entropy_bits(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def per_genome_summary(meta: pd.DataFrame, labels: np.ndarray, K: int) -> pd.DataFrame:
    df = meta.copy()
    df["label"] = labels
    rows = []
    for genome, g in df.groupby("name"):
        g = g.sort_values("start")
        lab = g["label"].to_numpy(int)
        n = len(lab)
        _, counts = np.unique(lab, return_counts=True)
        raw_h = entropy_bits(lab)
        switches = int(np.sum(lab[1:] != lab[:-1]) + (lab[-1] != lab[0])) if n > 1 else 0
        rows.append({
            "genome": genome,
            "n_windows": n,
            "dominant_fraction": float(counts.max() / n),
            "label_entropy_bits": raw_h,
            "label_entropy_normalized_log2K": float(raw_h / np.log2(K)) if K > 1 else 0.0,
            "switch_rate_circular": float(switches / n) if n else 0.0,
            "n_labels_observed": int(len(counts)),
        })
    return pd.DataFrame(rows).sort_values("genome").reset_index(drop=True)


def pairwise_ari(label_sets: List[np.ndarray]) -> Tuple[float, float, float]:
    vals = [adjusted_rand_score(a, b) for a, b in itertools.combinations(label_sets, 2)]
    if not vals:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, float(np.min(vals))


def safe_silhouette(X: np.ndarray, labels: np.ndarray, metric: str, sample_size: int, seed: int) -> float:
    size = min(sample_size, len(X)) if sample_size > 0 else None
    return float(silhouette_score(X, labels, metric=metric, sample_size=size, random_state=seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--kmer_k", type=int, default=4)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--k_values", default="2-12")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--baseline_k", type=int, default=8)
    ap.add_argument("--baseline_assignments", default=None, help="Optional core window_assignments.csv for baseline reproduction check")
    ap.add_argument("--silhouette_sample", type=int, default=3000)
    ap.add_argument("--euclidean_n_init", type=int, default=20)
    ap.add_argument("--spherical_n_init", type=int, default=5)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from pipeline.cssv_gb_pipeline import load_genomes, window_vectors

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exts = (".gb", ".gbk", ".genbank", ".fa", ".fasta", ".fna", ".txt")
    genomes = load_genomes(Path(args.input_dir).resolve(), exts=exts)
    meta, X, vocab = window_vectors(
        genomes, window=args.window, step=args.step, k=args.kmer_k, circular=True
    )
    Xn = normalize(X, norm="l2", axis=1, copy=True)
    k_values = parse_ints(args.k_values)
    seeds = parse_ints(args.seeds)
    metric_rows = []
    assignments = {}

    for K in k_values:
        for seed in seeds:
            model = KMeans(n_clusters=K, n_init=args.euclidean_n_init, random_state=seed)
            lab_e = model.fit_predict(X)
            assignments[("euclidean", K, seed)] = lab_e
            metric_rows.append({
                "geometry": "euclidean_raw_frequency",
                "K": K,
                "seed": seed,
                "objective": float(model.inertia_),
                "objective_name": "within_cluster_sum_squared_euclidean",
                "silhouette": safe_silhouette(X, lab_e, "euclidean", args.silhouette_sample, seed),
                "silhouette_metric": "euclidean",
                "calinski_harabasz": float(calinski_harabasz_score(X, lab_e)),
                "davies_bouldin": float(davies_bouldin_score(X, lab_e)),
                "iterations": int(model.n_iter_),
            })

            lab_s, loss_s, it_s = spherical_kmeans(
                X, K, random_state=seed, n_init=args.spherical_n_init
            )
            assignments[("spherical", K, seed)] = lab_s
            metric_rows.append({
                "geometry": "spherical_cosine",
                "K": K,
                "seed": seed,
                "objective": float(loss_s),
                "objective_name": "sum_one_minus_max_cosine_similarity",
                "silhouette": safe_silhouette(Xn, lab_s, "cosine", args.silhouette_sample, seed),
                "silhouette_metric": "cosine",
                "calinski_harabasz": float(calinski_harabasz_score(Xn, lab_s)),
                "davies_bouldin": float(davies_bouldin_score(Xn, lab_s)),
                "iterations": int(it_s),
            })

    seed_metrics = pd.DataFrame(metric_rows)
    seed_metrics.to_csv(out / "barcode_k_seed_metrics.csv", index=False)

    summary_rows = []
    for geometry_key, geometry_name in [("euclidean", "euclidean_raw_frequency"), ("spherical", "spherical_cosine")]:
        for K in k_values:
            labs = [assignments[(geometry_key, K, seed)] for seed in seeds]
            ari_mean, ari_sd, ari_min = pairwise_ari(labs)
            sub = seed_metrics[(seed_metrics.geometry == geometry_name) & (seed_metrics.K == K)]
            summary_rows.append({
                "geometry": geometry_name,
                "K": K,
                "n_seeds": len(seeds),
                "silhouette_mean": float(sub.silhouette.mean()),
                "silhouette_sd": float(sub.silhouette.std(ddof=1)),
                "calinski_harabasz_mean": float(sub.calinski_harabasz.mean()),
                "davies_bouldin_mean": float(sub.davies_bouldin.mean()),
                "objective_mean": float(sub.objective.mean()),
                "pairwise_seed_ARI_mean": ari_mean,
                "pairwise_seed_ARI_sd": ari_sd,
                "pairwise_seed_ARI_min": ari_min,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "barcode_k_selection_and_stability_summary.csv", index=False)

    best_rows = []
    for geometry, g in summary.groupby("geometry"):
        for metric, direction in [
            ("silhouette_mean", "max"),
            ("calinski_harabasz_mean", "max"),
            ("davies_bouldin_mean", "min"),
            ("pairwise_seed_ARI_mean", "max"),
        ]:
            idx = g[metric].idxmax() if direction == "max" else g[metric].idxmin()
            row = g.loc[idx]
            best_rows.append({"geometry": geometry, "criterion": metric, "direction": direction, "best_K": int(row.K), "value": float(row[metric])})
    pd.DataFrame(best_rows).to_csv(out / "barcode_best_k_by_criterion.csv", index=False)

    K = args.baseline_k
    seed = seeds[0]
    lab_e = assignments[("euclidean", K, seed)]
    lab_s = assignments[("spherical", K, seed)]
    meta_e = meta.copy(); meta_e["label"] = lab_e
    meta_s = meta.copy(); meta_s["label"] = lab_s
    meta_e.to_csv(out / f"window_assignments_euclidean_k{K}.csv", index=False)
    meta_s.to_csv(out / f"window_assignments_spherical_k{K}.csv", index=False)
    per_e = per_genome_summary(meta, lab_e, K).add_suffix("_euclidean").rename(columns={"genome_euclidean": "genome"})
    per_s = per_genome_summary(meta, lab_s, K).add_suffix("_spherical").rename(columns={"genome_spherical": "genome"})
    per_cmp = per_e.merge(per_s, on="genome")
    per_cmp.to_csv(out / f"per_genome_geometry_comparison_k{K}.csv", index=False)
    corr_rows = []
    for metric in ["dominant_fraction", "label_entropy_bits", "label_entropy_normalized_log2K", "switch_rate_circular"]:
        a = per_cmp[f"{metric}_euclidean"]
        b = per_cmp[f"{metric}_spherical"]
        corr_rows.append({"metric": metric, "spearman_rho": float(spearmanr(a, b).statistic)})
    pd.DataFrame([{
        "K": K,
        "seed": seed,
        "window_assignment_ARI": float(adjusted_rand_score(lab_e, lab_s)),
        "window_assignment_NMI": float(normalized_mutual_info_score(lab_e, lab_s)),
        "n_windows": len(meta),
        "n_features": X.shape[1],
        "circular_windows": True,
    }]).to_csv(out / f"geometry_assignment_comparison_k{K}.csv", index=False)
    if args.baseline_assignments:
        baseline_path = Path(args.baseline_assignments).resolve()
        base = pd.read_csv(baseline_path)
        needed = {"name", "start", "label"}
        if not needed.issubset(base.columns):
            raise ValueError(f"Baseline assignments must contain {sorted(needed)}")
        predicted = meta[["name", "start"]].copy()
        predicted["label_recomputed"] = lab_e
        cmp = predicted.merge(
            base[["name", "start", "label"]].rename(columns={"label": "label_core"}),
            on=["name", "start"], how="outer", indicator=True,
        )
        matched = cmp[cmp["_merge"] == "both"].copy()
        if matched.empty:
            raise ValueError("No overlapping name/start rows with baseline assignments")
        pd.DataFrame([{
            "baseline_assignments": str(baseline_path),
            "n_rows_recomputed": len(predicted),
            "n_rows_core": len(base),
            "n_rows_matched": len(matched),
            "n_rows_unmatched": int((cmp["_merge"] != "both").sum()),
            "assignment_ARI": float(adjusted_rand_score(matched["label_core"], matched["label_recomputed"])),
            "assignment_NMI": float(normalized_mutual_info_score(matched["label_core"], matched["label_recomputed"])),
            "exact_label_fraction": float(np.mean(matched["label_core"].to_numpy(int) == matched["label_recomputed"].to_numpy(int))),
        }]).to_csv(out / "baseline_assignment_reproduction_check.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / f"geometry_per_genome_metric_correlations_k{K}.csv", index=False)
    (out / "kmer_vocabulary.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    print(f"[OK] K diagnostics written to {out}")


if __name__ == "__main__":
    main()
