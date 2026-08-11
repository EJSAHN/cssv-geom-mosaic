#!/usr/bin/env python3
"""Validate whole-genome k-mer structure against published species labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder


def read_distance(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, index_col=0)
    d.index = d.index.astype(str).str.strip()
    d.columns = pd.Index([str(x).strip() for x in d.columns])
    if set(d.index) != set(d.columns):
        raise ValueError("Distance-matrix row and column labels differ")
    d = d.loc[d.index, d.index].apply(pd.to_numeric, errors="coerce")
    arr = d.to_numpy(float)
    if not np.isfinite(arr).all():
        raise ValueError("Distance matrix contains non-finite values")
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=d.index, columns=d.index)


def agglomerative_precomputed(arr: np.ndarray, n_clusters: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average"
        )
    return model.fit_predict(arr)


def best_cluster_to_species(species: pd.Series, clusters: np.ndarray) -> dict[int, str]:
    table = pd.crosstab(
        pd.Series(np.asarray(clusters), name="cluster"),
        pd.Series(species.to_numpy(), name="species"),
    )
    # Maximize matched genomes using the Hungarian algorithm.
    cost = table.to_numpy().max() - table.to_numpy()
    rows, cols = linear_sum_assignment(cost)
    return {int(table.index[r]): str(table.columns[c]) for r, c in zip(rows, cols)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance_matrix", required=True)
    ap.add_argument("--species_table", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_clusters", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = read_distance(Path(args.distance_matrix).resolve())
    sp = pd.read_csv(Path(args.species_table).resolve(), sep="\t", dtype=str)
    required = {"accession", "species_acronym", "published_virus_name"}
    if not required.issubset(sp.columns):
        raise ValueError(f"Species table must contain {sorted(required)}")
    sp["accession"] = sp["accession"].astype(str).str.strip()
    sp = sp.drop_duplicates("accession").set_index("accession")

    missing = sorted(set(d.index) - set(sp.index))
    extra = sorted(set(sp.index) - set(d.index))
    if missing:
        raise ValueError(f"Missing published species labels for: {missing}")
    sp = sp.loc[d.index].copy()
    sp.index.name = "accession"

    n_clusters = args.n_clusters or int(sp["species_acronym"].nunique())
    clusters = agglomerative_precomputed(d.to_numpy(float), n_clusters=n_clusters)
    y = LabelEncoder().fit_transform(sp["species_acronym"])
    ari = float(adjusted_rand_score(y, clusters))
    nmi = float(normalized_mutual_info_score(y, clusters))
    sil = float(silhouette_score(d.to_numpy(float), clusters, metric="precomputed"))

    mapping = best_cluster_to_species(sp["species_acronym"], clusters)
    result = sp.reset_index().copy()
    result["kmer_cluster"] = clusters
    result["cluster_mapped_species"] = [mapping.get(int(c), "unmapped") for c in clusters]
    result["species_match"] = result["species_acronym"] == result["cluster_mapped_species"]
    result.to_csv(out / "external_species_cluster_mapping.csv", index=False)

    contingency = pd.crosstab(
        pd.Series(sp["species_acronym"].to_numpy(), name="published_species"),
        pd.Series(clusters, name="kmer_cluster"),
    )
    contingency.to_csv(out / "external_species_contingency.csv")

    summary = pd.DataFrame([
        {
            "n_genomes": len(d),
            "n_published_species": int(sp["species_acronym"].nunique()),
            "n_kmer_clusters": int(n_clusters),
            "clustering_method": "average-linkage agglomerative clustering",
            "distance": "whole-genome k-mer cosine distance",
            "adjusted_rand_index": ari,
            "normalized_mutual_information": nmi,
            "silhouette_precomputed_distance": sil,
            "n_species_matches_after_optimal_mapping": int(result["species_match"].sum()),
            "n_species_mismatches_after_optimal_mapping": int((~result["species_match"]).sum()),
            "missing_species_labels": ";".join(missing),
            "unused_species_table_accessions": ";".join(extra),
        }
    ])
    summary.to_csv(out / "external_species_validation_summary.csv", index=False)
    result.loc[:, [
        "accession", "species_acronym", "published_virus_name",
        "assignment_source", "assignment_basis"
    ]].to_csv(out / "published_species_assignments_core48.csv", index=False)

    print(f"[OK] ARI={ari:.6f}; NMI={nmi:.6f}; silhouette={sil:.6f}")
    print(f"[OK] Outputs: {out}")


if __name__ == "__main__":
    main()
