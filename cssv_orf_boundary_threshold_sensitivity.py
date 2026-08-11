#!/usr/bin/env python3
"""Sensitivity of ORF-boundary enrichment to the distance threshold."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def parse_ints(text: str) -> List[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or any(x < 0 for x in vals):
        raise ValueError("thresholds must be non-negative comma-separated integers")
    return vals


def circular_distance_to_boundaries(pos: np.ndarray, boundaries: np.ndarray, length: int) -> np.ndarray:
    pos = np.asarray(pos, dtype=int)[:, None]
    b = np.asarray(boundaries, dtype=int)[None, :] % int(length)
    direct = np.abs(pos - b)
    return np.min(np.minimum(direct, int(length) - direct), axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome_summary", required=True)
    ap.add_argument("--predicted_orfs", required=True)
    ap.add_argument("--switchpoints", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--thresholds", default="100,200,400")
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    thresholds = parse_ints(args.thresholds)
    gs = pd.read_csv(args.genome_summary)
    orfs = pd.read_csv(args.predicted_orfs)
    sw = pd.read_csv(args.switchpoints)

    name_col = "name" if "name" in gs.columns else "genome"
    length_map = dict(zip(gs[name_col].astype(str), pd.to_numeric(gs["length"], errors="raise").astype(int)))
    orf_name = "name" if "name" in orfs.columns else "genome"
    sw_name = "name" if "name" in sw.columns else "genome"
    boundaries: Dict[str, np.ndarray] = {}
    for name, g in orfs.groupby(orf_name):
        vals = pd.concat([g["start0"], g["end0"]], ignore_index=True)
        boundaries[str(name)] = pd.to_numeric(vals, errors="coerce").dropna().astype(int).to_numpy()
    switch_positions = {
        str(name): pd.to_numeric(g["pos0"], errors="coerce").dropna().astype(int).to_numpy()
        for name, g in sw.groupby(sw_name)
    }

    observed_distances = []
    for name, positions in switch_positions.items():
        if name not in length_map or name not in boundaries or boundaries[name].size == 0:
            continue
        observed_distances.extend(
            circular_distance_to_boundaries(positions % length_map[name], boundaries[name], length_map[name]).tolist()
        )
    observed_distances = np.asarray(observed_distances, dtype=float)
    if observed_distances.size == 0:
        raise ValueError("No valid switchpoint-to-boundary distances could be computed")

    rng = np.random.default_rng(args.seed)
    null = np.zeros((args.perm, len(thresholds)), dtype=float)
    for r in range(args.perm):
        all_d = []
        for name, obs_pos in switch_positions.items():
            if name not in length_map or name not in boundaries or boundaries[name].size == 0:
                continue
            L = length_map[name]
            rand_pos = rng.integers(0, L, size=len(obs_pos), endpoint=False)
            all_d.extend(circular_distance_to_boundaries(rand_pos, boundaries[name], L).tolist())
        all_d = np.asarray(all_d, dtype=float)
        for j, threshold in enumerate(thresholds):
            null[r, j] = float(np.mean(all_d <= threshold))

    rows = []
    for j, threshold in enumerate(thresholds):
        obs_frac = float(np.mean(observed_distances <= threshold))
        vals = null[:, j]
        p = float((1 + np.sum(vals >= obs_frac)) / (len(vals) + 1))
        sd = float(np.std(vals, ddof=1))
        z = float((obs_frac - np.mean(vals)) / sd) if sd > 0 else np.nan
        rows.append({
            "threshold_bp": threshold,
            "total_switchpoints": int(observed_distances.size),
            "observed_fraction_near_boundary": obs_frac,
            "null_mean_fraction": float(np.mean(vals)),
            "null_sd_fraction": sd,
            "empirical_p_one_sided_enrichment": p,
            "z_score": z,
            "permutations": int(args.perm),
            "seed": int(args.seed),
            "test_direction_pre_specified": "enrichment; Pr(null fraction >= observed fraction)",
            "circular_distance": True,
        })
    pd.DataFrame(rows).to_csv(out / "orf_boundary_threshold_sensitivity_summary.csv", index=False)
    null_df = pd.DataFrame(null, columns=[f"threshold_{x}bp" for x in thresholds])
    null_df.insert(0, "permutation", np.arange(1, args.perm + 1))
    null_df.to_csv(out / "orf_boundary_threshold_null_fractions.csv", index=False)
    pd.DataFrame({"observed_distance_bp": observed_distances}).to_csv(
        out / "observed_switchpoint_orf_boundary_distances.csv", index=False
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
