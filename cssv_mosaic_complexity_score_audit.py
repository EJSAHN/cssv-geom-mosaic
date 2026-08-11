#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the composite mosaic-complexity score.

The score is defined as:

    (1 - dominant-label purity) + normalized barcode entropy + switch rate

Raw barcode entropy is measured in bits (log base 2) and normalized by
``log2(mosaic_k)`` when a normalized value is not already present.

ORF–mosaic mismatch is retained in the merged input table as an independent
diagnostic, but it is not included in the mosaic-complexity score.

Inputs
------
--merged
    ``mosaic_orf_merged_per_genome.csv`` from
    ``pipeline/cssv_tree_mosaic_agreement.py``.

Outputs
-------
- ``mosaic_complexity_score_formula_components.csv``
- ``mosaic_complexity_score_distribution_summary.csv``
- ``mosaic_complexity_score_weight_sensitivity.csv``
- ``mosaic_complexity_score_topN_by_weight_scheme.csv``

The utility writes CSV files only and has no hard-coded project paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input table is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def compute_components(df: pd.DataFrame, mosaic_k: int = 8) -> pd.DataFrame:
    ensure_columns(df, ["genome", "dominant_frac", "switch_rate"])
    out = df.copy()
    out["dominant_frac"] = pd.to_numeric(out["dominant_frac"], errors="coerce")
    out["switch_rate"] = pd.to_numeric(out["switch_rate"], errors="coerce")

    if "norm_entropy" in out.columns:
        out["norm_entropy"] = pd.to_numeric(out["norm_entropy"], errors="coerce")
    elif "label_entropy_bits" in out.columns or "label_entropy" in out.columns:
        ent_col = "label_entropy_bits" if "label_entropy_bits" in out.columns else "label_entropy"
        ent = pd.to_numeric(out[ent_col], errors="coerce")
        if int(mosaic_k) <= 1:
            raise ValueError("mosaic_k must be greater than 1")
        out["norm_entropy"] = ent / np.log2(int(mosaic_k))
        out["entropy_normalization_k"] = int(mosaic_k)
        out["entropy_log_base"] = 2
    else:
        raise ValueError("Input table must contain 'norm_entropy', 'label_entropy_bits', or 'label_entropy'.")

    if (out["norm_entropy"] > 1.0 + 1e-10).any():
        raise ValueError("Normalized entropy exceeded 1; verify mosaic_k and input labels.")

    out["component_low_purity"] = 1.0 - out["dominant_frac"]
    out["component_entropy"] = out["norm_entropy"]
    out["component_switch_rate"] = out["switch_rate"]
    out["mosaic_complexity_score_recomputed"] = (
        out["component_low_purity"]
        + out["component_entropy"]
        + out["component_switch_rate"]
    )

    if "mosaic_complexity_score" in out.columns:
        out["mosaic_complexity_score_original"] = pd.to_numeric(
            out["mosaic_complexity_score"], errors="coerce"
        )
        out["mosaic_complexity_score_delta"] = (
            out["mosaic_complexity_score_recomputed"]
            - out["mosaic_complexity_score_original"]
        )
    return out


def score_with_weights(df: pd.DataFrame, weights: Tuple[float, float, float]) -> pd.Series:
    wp, we, ws = weights
    return (
        wp * df["component_low_purity"]
        + we * df["component_entropy"]
        + ws * df["component_switch_rate"]
    )


def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit the mosaic-complexity score formula, distribution, and weight sensitivity."
    )
    ap.add_argument("--merged", required=True, help="mosaic_orf_merged_per_genome.csv")
    ap.add_argument("--out_dir", default="results/mosaic_complexity_score_audit")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--mosaic_k", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged)
    audit = compute_components(df, mosaic_k=args.mosaic_k)
    audit = audit.sort_values(
        ["mosaic_complexity_score_recomputed", "dominant_frac", "switch_rate", "genome"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    audit["rank_recomputed"] = np.arange(1, len(audit) + 1)

    component_cols = [
        "rank_recomputed", "genome", "dominant_frac", "norm_entropy", "switch_rate",
        "component_low_purity", "component_entropy", "component_switch_rate",
        "mosaic_complexity_score_recomputed",
    ]
    if "mosaic_complexity_score_original" in audit.columns:
        component_cols += ["mosaic_complexity_score_original", "mosaic_complexity_score_delta"]
    for optional in [
        "dominant_label", "label_entropy_bits", "label_entropy", "n_labels_observed",
        "n_labels", "n_switches", "orf_cluster", "mapped_mosaic_from_orf", "mismatch",
    ]:
        if optional in audit.columns and optional not in component_cols:
            component_cols.append(optional)
    audit[component_cols].to_csv(
        out_dir / "mosaic_complexity_score_formula_components.csv", index=False
    )

    scores = audit["mosaic_complexity_score_recomputed"].to_numpy(float)
    top_n = int(args.top_n)
    top_genomes = audit.head(top_n)["genome"].tolist()
    rank_gap = np.nan
    if len(scores) > top_n:
        rank_gap = float(scores[top_n - 1] - scores[top_n])

    pd.DataFrame([
        {
            "n_genomes": len(audit),
            "top_n": top_n,
            "score_min": float(np.nanmin(scores)),
            "score_q25": float(np.nanpercentile(scores, 25)),
            "score_median": float(np.nanmedian(scores)),
            "score_q75": float(np.nanpercentile(scores, 75)),
            "score_max": float(np.nanmax(scores)),
            "score_rank_topN": float(scores[top_n - 1]) if len(scores) >= top_n else np.nan,
            "score_rank_topN_plus1": float(scores[top_n]) if len(scores) > top_n else np.nan,
            "gap_topN_vs_next": rank_gap,
            "topN_genomes": ";".join(top_genomes),
        }
    ]).to_csv(out_dir / "mosaic_complexity_score_distribution_summary.csv", index=False)

    weight_schemes: Dict[str, Tuple[float, float, float]] = {
        "equal": (1.0, 1.0, 1.0),
        "no_switch": (1.0, 1.0, 0.0),
        "no_entropy": (1.0, 0.0, 1.0),
        "double_low_purity": (2.0, 1.0, 1.0),
        "double_entropy": (1.0, 2.0, 1.0),
        "double_switch": (1.0, 1.0, 2.0),
        "half_low_purity": (0.5, 1.0, 1.0),
        "half_entropy": (1.0, 0.5, 1.0),
        "half_switch": (1.0, 1.0, 0.5),
    }

    baseline_score = score_with_weights(audit, weight_schemes["equal"])
    baseline_rank = audit.assign(_score=baseline_score).sort_values(
        ["_score", "genome"], ascending=[False, True]
    )["genome"].tolist()
    baseline_top = baseline_rank[:top_n]

    sens_rows = []
    top_rows = []
    for name, weights in weight_schemes.items():
        sc = score_with_weights(audit, weights)
        tmp = audit[["genome"]].copy()
        tmp["score"] = sc
        tmp = tmp.sort_values(["score", "genome"], ascending=[False, True]).reset_index(drop=True)
        tmp["rank"] = np.arange(1, len(tmp) + 1)
        top = tmp.head(top_n)["genome"].tolist()
        rho = np.nan
        if spearmanr is not None:
            rho = float(spearmanr(baseline_score, sc, nan_policy="omit").correlation)
        sens_rows.append({
            "scheme": name,
            "w_low_purity": weights[0],
            "w_entropy": weights[1],
            "w_switch": weights[2],
            "topN_overlap_count": len(set(baseline_top) & set(top)),
            "topN_jaccard_vs_equal": jaccard(baseline_top, top),
            "spearman_score_vs_equal": rho,
            "topN_genomes": ";".join(top),
        })
        for _, row in tmp.head(top_n).iterrows():
            top_rows.append({
                "scheme": name,
                "rank": int(row["rank"]),
                "genome": row["genome"],
                "score": float(row["score"]),
            })

    pd.DataFrame(sens_rows).to_csv(
        out_dir / "mosaic_complexity_score_weight_sensitivity.csv", index=False
    )
    pd.DataFrame(top_rows).to_csv(
        out_dir / "mosaic_complexity_score_topN_by_weight_scheme.csv", index=False
    )

    print("[OK] Wrote mosaic-complexity score audit outputs to", out_dir)


if __name__ == "__main__":
    main()
