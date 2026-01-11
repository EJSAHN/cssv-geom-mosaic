"""
Make a paper-ready Table 1 for top-N chimera candidates.

Inputs:
- chimera_candidates.csv (ranked)
- mosaic_orf_merged_per_genome.csv (full metrics per genome)

Outputs:
- chimera_table_topN.csv
- chimera_table_topN.md (markdown table)
- chimera_table_topN.tex (simple LaTeX tabular)

Usage:
python scripts/cssv_make_chimera_table.py ^
  --chimera "results/tree_mosaic_agreement/chimera_candidates.csv" ^
  --merged "results/tree_mosaic_agreement/mosaic_orf_merged_per_genome.csv" ^
  --out_dir "results/tree_mosaic_agreement" ^
  --top_n 10
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


DEFAULT_COLS = [
    "genome",
    "chimera_score",
    "dominant_label",
    "dominant_frac",
    "n_labels",
    "label_entropy",
    "switch_rate",
    "n_switches",
    "orf_cluster",
    "mapped_mosaic_from_orf",
    "mismatch",
]


def to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def to_latex(df: pd.DataFrame, caption: str = "Top chimera candidates", label: str = "tab:chimera") -> str:
    # simple LaTeX table (no booktabs dependency)
    cols = df.columns.tolist()
    header = " & ".join(cols) + r" \\"
    lines = [r"\begin{table}[ht]", r"\centering", r"\small", r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline", header, r"\hline"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chimera", required=True, help="chimera_candidates.csv")
    ap.add_argument("--merged", required=True, help="mosaic_orf_merged_per_genome.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--cols", default=None, help="Comma-separated columns to include (optional)")
    args = ap.parse_args()

    chim = pd.read_csv(args.chimera)
    merged = pd.read_csv(args.merged)

    if "genome" not in chim.columns or "genome" not in merged.columns:
        raise ValueError("Both chimera and merged files must have 'genome' column.")

    top = chim.head(int(args.top_n))["genome"].astype(str).tolist()
    m = merged.copy()
    m["genome"] = m["genome"].astype(str)

    df = m[m["genome"].isin(top)].copy()

    # Keep in the same ranked order as chimera_candidates
    rank_map = {g: i + 1 for i, g in enumerate(top)}
    df["rank"] = df["genome"].map(rank_map).astype(int)
    df = df.sort_values("rank")

    cols = DEFAULT_COLS
    if args.cols:
        cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    keep_cols = ["rank"] + [c for c in cols if c in df.columns]
    df_out = df[keep_cols].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"chimera_table_top{args.top_n}.csv"
    md_path = out_dir / f"chimera_table_top{args.top_n}.md"
    tex_path = out_dir / f"chimera_table_top{args.top_n}.tex"

    df_out.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df_out), encoding="utf-8")
    tex_path.write_text(to_latex(df_out, caption=f"Top {args.top_n} chimera candidates"), encoding="utf-8")

    print("Done.")
    print(f"- {csv_path}")
    print(f"- {md_path}")
    print(f"- {tex_path}")


if __name__ == "__main__":
    main()
