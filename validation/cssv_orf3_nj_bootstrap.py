#!/usr/bin/env python3
"""Bootstrap support for an unrooted ORF3 neighbor-joining tree.

Alignment columns are resampled with replacement. Pairwise identity distances
are recalculated for each replicate, NJ trees are reconstructed, and support is
reported for non-trivial unrooted bipartitions. No biological outgroup is used.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from Bio import AlignIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor


def clean_names(aln, split: str | None) -> Tuple[List[str], np.ndarray]:
    names = []
    seqs = []
    for rec in aln:
        name = str(rec.id).strip()
        if split:
            name = name.split(split)[0]
        names.append(name)
        seqs.append(np.frombuffer(str(rec.seq).upper().encode("ascii"), dtype="S1"))
    if len(names) != len(set(names)):
        raise ValueError("Sequence names are not unique after name splitting")
    return names, np.vstack(seqs)


def pair_arrays(arr: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    n, L = arr.shape
    pairs: List[Tuple[int, int]] = []
    valid_rows = []
    match_rows = []
    invalid = np.isin(arr, np.array([b"-", b".", b"X", b"?"], dtype="S1"))
    for i in range(n):
        for j in range(i + 1, n):
            valid = ~(invalid[i] | invalid[j])
            match = valid & (arr[i] == arr[j])
            pairs.append((i, j))
            valid_rows.append(valid.astype(np.float32))
            match_rows.append(match.astype(np.float32))
    return pairs, np.vstack(valid_rows), np.vstack(match_rows)


def distance_from_weights(
    names: List[str],
    pairs: List[Tuple[int, int]],
    valid: np.ndarray,
    match: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    compared = valid @ weights
    matched = match @ weights
    distances = np.divide(
        matched, compared, out=np.full_like(matched, np.nan, dtype=float), where=compared > 0
    )
    distances = 1.0 - distances
    n = len(names)
    matrix = np.zeros((n, n), dtype=float)
    for value, (i, j) in zip(distances, pairs):
        if not np.isfinite(value):
            raise ValueError(f"No comparable alignment columns for pair {names[i]}, {names[j]}")
        matrix[i, j] = matrix[j, i] = float(value)
    return pd.DataFrame(matrix, index=names, columns=names)


def nj_tree(dist: pd.DataFrame):
    names = list(dist.index)
    lower = [[float(dist.iat[i, j]) for j in range(i + 1)] for i in range(len(names))]
    tree = DistanceTreeConstructor().nj(DistanceMatrix(names, lower))
    tree.rooted = False
    return tree


def canonical_split(side: Iterable[str], all_taxa: FrozenSet[str]) -> FrozenSet[str]:
    a = frozenset(side)
    b = all_taxa - a
    if len(a) < len(b):
        return a
    if len(b) < len(a):
        return b
    return min(a, b, key=lambda x: tuple(sorted(x)))


def tree_splits(tree, taxa: FrozenSet[str]) -> Set[FrozenSet[str]]:
    out: Set[FrozenSet[str]] = set()
    for clade in tree.find_clades(order="preorder"):
        if clade is tree.root or clade.is_terminal():
            continue
        desc = frozenset(str(x.name) for x in clade.get_terminals())
        split = canonical_split(desc, taxa)
        if 1 < len(split) < len(taxa) - 1:
            out.add(split)
    return out


def annotate_support(tree, support: Dict[FrozenSet[str], float], taxa: FrozenSet[str]) -> None:
    for clade in tree.find_clades(order="preorder"):
        if clade is tree.root or clade.is_terminal():
            continue
        desc = frozenset(str(x.name) for x in clade.get_terminals())
        split = canonical_split(desc, taxa)
        if split in support:
            clade.confidence = round(float(support[split]), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", required=True, help="Trimmed ORF3 amino-acid alignment")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name_split", default="|", help="Use 'none' to retain complete sequence IDs")
    ap.add_argument("--report_support_min", type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    aln = AlignIO.read(args.alignment, "fasta")
    split = None if args.name_split.lower() == "none" else args.name_split
    names, arr = clean_names(aln, split)
    n, L = arr.shape
    if n < 4 or L < 1:
        raise ValueError("Alignment is too small for bootstrap analysis")
    pairs, valid, match = pair_arrays(arr)
    original_weights = np.ones(L, dtype=np.float32)
    ref_dist = distance_from_weights(names, pairs, valid, match, original_weights)
    ref_dist.to_csv(out / "orf3_identity_distance_from_alignment.csv")
    reference_tree = nj_tree(ref_dist)
    taxa = frozenset(names)
    ref_splits = tree_splits(reference_tree, taxa)
    counts = {s: 0 for s in ref_splits}

    rng = np.random.default_rng(args.seed)
    probabilities = np.full(L, 1.0 / L)
    for replicate in range(1, args.replicates + 1):
        weights = rng.multinomial(L, probabilities).astype(np.float32)
        dist = distance_from_weights(names, pairs, valid, match, weights)
        splits = tree_splits(nj_tree(dist), taxa)
        for s in ref_splits & splits:
            counts[s] += 1
        if replicate % max(1, args.replicates // 10) == 0:
            print(f"[INFO] bootstrap {replicate}/{args.replicates}")

    support = {s: 100.0 * counts[s] / args.replicates for s in ref_splits}
    annotate_support(reference_tree, support, taxa)
    Phylo.write(reference_tree, str(out / "orf3_nj_bootstrap_unrooted.newick"), "newick")
    display_tree = copy.deepcopy(reference_tree)
    try:
        display_tree.root_at_midpoint()
        display_tree.rooted = True
        Phylo.write(display_tree, str(out / "orf3_nj_bootstrap_midpoint_display.newick"), "newick")
    except Exception as exc:
        print(f"[WARN] Midpoint rooting failed: {exc}")

    rows = []
    for split_side, value in sorted(support.items(), key=lambda kv: (-kv[1], len(kv[0]), sorted(kv[0]))):
        if value < args.report_support_min:
            continue
        rows.append({
            "split_size_smaller_side": len(split_side),
            "smaller_side_taxa": ";".join(sorted(split_side)),
            "bootstrap_support_percent": value,
            "supporting_replicates": counts[split_side],
            "total_replicates": args.replicates,
        })
    pd.DataFrame(rows).to_csv(out / "orf3_nj_bootstrap_support.csv", index=False)
    pd.DataFrame([{
        "n_taxa": n,
        "alignment_columns": L,
        "bootstrap_replicates": args.replicates,
        "seed": args.seed,
        "tree_method": "neighbor joining on pairwise amino-acid identity distance",
        "rooting_for_inference": "unrooted; no outgroup",
        "display_tree": "midpoint rooted for display only",
        "n_nontrivial_reference_splits": len(ref_splits),
        "n_splits_support_ge_70": int(sum(v >= 70 for v in support.values())),
        "n_splits_support_ge_90": int(sum(v >= 90 for v in support.values())),
    }]).to_csv(out / "orf3_nj_bootstrap_summary.csv", index=False)
    print(f"[OK] Bootstrap outputs: {out}")


if __name__ == "__main__":
    main()
