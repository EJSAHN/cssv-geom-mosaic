#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cssv_mosaic_discordance_score_audit.py

Score-audit utility for the CSSV genome-geometry/mosaic-barcode pipeline.

Purpose
-------
  * how the composite mosaic-discordance score is computed,
  * whether the top-ranked genomes are separated from the rest or form a continuum,
  * whether rankings are robust to simple weight perturbations.

Inputs
------
  --merged  mosaic_orf_merged_per_genome.csv from cssv_tree_mosaic_agreement.py

Outputs
-------
  mosaic_discordance_score_formula_components.csv
  mosaic_discordance_score_distribution_summary.csv
  mosaic_discordance_score_weight_sensitivity.csv
  mosaic_discordance_score_topN_by_weight_scheme.csv

No hardcoded directories. Suitable for GitHub.
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


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}. Available columns: {list(df.columns)}")


def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(df, ["genome", "dominant_frac", "switch_rate", "mismatch"])
    out = df.copy()
    out["dominant_frac"] = pd.to_numeric(out["dominant_frac"], errors="coerce")
    out["switch_rate"] = pd.to_numeric(out["switch_rate"], errors="coerce")
    out["mismatch_bool"] = parse_bool_series(out["mismatch"])

    if "norm_entropy" in out.columns:
        out["norm_entropy"] = pd.to_numeric(out["norm_entropy"], errors="coerce")
    elif "label_entropy" in out.columns:
        # Conservative fallback if old output lacks norm_entropy.
        # Use the number of observed dominant labels as denominator.
        ent = pd.to_numeric(out["label_entropy"], errors="coerce")
        if "dominant_label" in out.columns:
            n_labels = max(2, int(out["dominant_label"].nunique()))
        else:
            n_labels = max(2, int(np.ceil(np.nanmax(ent))) if np.isfinite(np.nanmax(ent)) else 8)
        out["norm_entropy"] = ent / np.log2(n_labels)
    else:
        raise ValueError("Input table must contain either 'norm_entropy' or 'label_entropy'.")

    out["component_low_purity"] = 1.0 - out["dominant_frac"]
    out["component_entropy"] = out["norm_entropy"]
    out["component_switch_rate"] = out["switch_rate"]
    out["component_mismatch"] = out["mismatch_bool"].astype(float)
    out["mosaic_discordance_score_recomputed"] = (
        out["component_low_purity"]
        + out["component_entropy"]
        + out["component_switch_rate"]
        + out["component_mismatch"]
    )

    original_score_col = None
    if "mosaic_discordance_score" in out.columns:
        original_score_col = "mosaic_discordance_score"
    if original_score_col is not None:
        out["mosaic_discordance_score_original"] = pd.to_numeric(out[original_score_col], errors="coerce")
        out["mosaic_discordance_score_delta"] = (
            out["mosaic_discordance_score_recomputed"] - out["mosaic_discordance_score_original"]
        )
    return out


def score_with_weights(df: pd.DataFrame, weights: Tuple[float, float, float, float]) -> pd.Series:
    wp, we, ws, wm = weights
    return (
        wp * df["component_low_purity"]
        + we * df["component_entropy"]
        + ws * df["component_switch_rate"]
        + wm * df["component_mismatch"]
    )


def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit mosaic-discordance score formula, distribution, and weight sensitivity.")
    ap.add_argument("--merged", required=True, help="mosaic_orf_merged_per_genome.csv")
    ap.add_argument("--out_dir", default="results/mosaic_discordance_score_audit", help="Output directory")
    ap.add_argument("--top_n", type=int, default=10, help="Top N used for overlap analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged)
    audit = compute_components(df)
    audit = audit.sort_values("mosaic_discordance_score_recomputed", ascending=False).reset_index(drop=True)
    audit["rank_recomputed"] = np.arange(1, len(audit) + 1)

    component_cols = [
        "rank_recomputed", "genome", "dominant_frac", "norm_entropy", "switch_rate", "mismatch_bool",
        "component_low_purity", "component_entropy", "component_switch_rate", "component_mismatch",
        "mosaic_discordance_score_recomputed",
    ]
    if "mosaic_discordance_score_original" in audit.columns:
        component_cols += ["mosaic_discordance_score_original", "mosaic_discordance_score_delta"]
    for optional in ["dominant_label", "label_entropy", "n_labels", "n_switches", "orf_cluster", "mapped_mosaic_from_orf"]:
        if optional in audit.columns and optional not in component_cols:
            component_cols.append(optional)
    audit[component_cols].to_csv(out_dir / "mosaic_discordance_score_formula_components.csv", index=False)

    scores = audit["mosaic_discordance_score_recomputed"].to_numpy(float)
    top_n = int(args.top_n)
    top_genomes = audit.head(top_n)["genome"].tolist()
    rank_gap = np.nan
    if len(scores) > top_n:
        rank_gap = float(scores[top_n - 1] - scores[top_n])

    summary = pd.DataFrame([
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
    ])
    summary.to_csv(out_dir / "mosaic_discordance_score_distribution_summary.csv", index=False)

    weight_schemes: Dict[str, Tuple[float, float, float, float]] = {
        "equal_original": (1.0, 1.0, 1.0, 1.0),
        "no_mismatch": (1.0, 1.0, 1.0, 0.0),
        "no_switch": (1.0, 1.0, 0.0, 1.0),
        "no_entropy": (1.0, 0.0, 1.0, 1.0),
        "double_low_purity": (2.0, 1.0, 1.0, 1.0),
        "double_entropy": (1.0, 2.0, 1.0, 1.0),
        "double_switch": (1.0, 1.0, 2.0, 1.0),
        "half_mismatch": (1.0, 1.0, 1.0, 0.5),
        "all_continuous_only": (1.0, 1.0, 1.0, 0.0),
    }

    baseline_score = score_with_weights(audit, weight_schemes["equal_original"])
    baseline_rank = audit.assign(_score=baseline_score).sort_values("_score", ascending=False)["genome"].tolist()
    baseline_top = baseline_rank[:top_n]

    sens_rows = []
    top_rows = []
    for name, weights in weight_schemes.items():
        sc = score_with_weights(audit, weights)
        tmp = audit[["genome"]].copy()
        tmp["score"] = sc
        tmp = tmp.sort_values("score", ascending=False).reset_index(drop=True)
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
            "w_mismatch": weights[3],
            "topN_overlap_count": len(set(baseline_top) & set(top)),
            "topN_jaccard_vs_equal": jaccard(baseline_top, top),
            "spearman_score_vs_equal": rho,
            "topN_genomes": ";".join(top),
        })
        for _, row in tmp.head(top_n).iterrows():
            top_rows.append({"scheme": name, "rank": int(row["rank"]), "genome": row["genome"], "score": float(row["score"])})

    pd.DataFrame(sens_rows).to_csv(out_dir / "mosaic_discordance_score_weight_sensitivity.csv", index=False)
    pd.DataFrame(top_rows).to_csv(out_dir / "mosaic_discordance_score_topN_by_weight_scheme.csv", index=False)


    print("[OK] Wrote mosaic-discordance score audit outputs to", out_dir)
    print("[OK] CSV outputs can be added to Supplementary Data S1.")


if __name__ == "__main__":
    main()
