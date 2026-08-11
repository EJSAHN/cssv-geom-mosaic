"""
Cleaner figure:
- Left: mosaic barcodes for top N highest-scoring genomes
- Right: ORF3 NJ tree, tips renamed to accession only
        selected genomes highlighted (red + trailing '*')

Outputs vector PDF + PNG 300dpi.

Usage:
python scripts/cssv_plot_mosaic_barcode_tree_clean.py ^
  --candidates "results/tree_mosaic_agreement/mosaic_ranked_genomes.csv" ^
  --window_assignments "results/gb/window_assignments.csv" ^
  --tree_newick "results/orf3_phylogeny/nj_tree.newick" ^
  --out_prefix "results/tree_mosaic_agreement/mosaic_top10_barcode_tree_clean" ^
  --top_n 10
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def parse_window_assignments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    if "genome" in cols:
        gcol = cols["genome"]
    elif "name" in cols:
        gcol = cols["name"]
    else:
        raise ValueError("window_assignments missing genome/name column")

    if "start" not in cols:
        raise ValueError("window_assignments missing start column")
    scol = cols["start"]

    if "label" in cols:
        lcol = cols["label"]
    elif "assigned_cluster" in cols:
        lcol = cols["assigned_cluster"]
    elif "cluster" in cols:
        lcol = cols["cluster"]
    else:
        raise ValueError("window_assignments missing label/assigned_cluster column")

    out = df[[gcol, scol, lcol]].copy()
    out.columns = ["genome", "start", "label"]
    out["genome"] = out["genome"].astype(str).str.strip()
    out["start"] = pd.to_numeric(out["start"], errors="coerce").fillna(0).astype(int)
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
    return out


def tip_to_accession(tip: str) -> str:
    s = str(tip).strip()
    # drop anything after '|'
    s = s.split("|")[0]
    # drop any duplicate suffixes
    s = s.split("__dup")[0]
    return s


def draw_barcode(ax, win: pd.DataFrame, genomes: list[str], title: str):
    cmap = plt.get_cmap("tab20")
    g_to_y = {g: i for i, g in enumerate(genomes)}
    sub = win[win["genome"].isin(genomes)].copy().sort_values(["genome", "start"])

    xs = sub["start"].to_numpy()
    ys = sub["genome"].map(g_to_y).astype(int).to_numpy()
    cs = sub["label"].to_numpy()

    sc = ax.scatter(xs, ys, c=cs, cmap=cmap, marker="s", s=18)
    ax.set_yticks(range(len(genomes)))
    ax.set_yticklabels(genomes, fontsize=7)
    ax.set_xlabel("Genome position (start bp)")
    ax.set_title(title)

    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Mosaic label")
    return ax


def draw_tree_clean(ax, newick_path: Path, highlight: set[str], title: str):
    from Bio import Phylo

    tree = Phylo.read(str(newick_path), "newick")

    # rename terminals to accession only (+ mark if highlighted)
    for term in tree.get_terminals():
        acc = tip_to_accession(term.name)
        term.name = f"{acc}*" if acc in highlight else acc

    Phylo.draw(tree, axes=ax, do_show=False)
    ax.set_title(title)

    # color highlighted tip labels (texts)
    for txt in ax.texts:
        t = txt.get_text().strip()
        acc = t[:-1] if t.endswith("*") else t
        if acc in highlight:
            txt.set_color("red")
            txt.set_fontweight("bold")

    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", "--mosaic", dest="candidates", required=True, help="ranked-genome CSV")
    ap.add_argument("--window_assignments", required=True)
    ap.add_argument("--tree_newick", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    ranked = pd.read_csv(args.candidates).copy()
    if "genome" not in ranked.columns:
        raise ValueError("ranked-genome CSV must have column 'genome'")
    ranked["genome"] = ranked["genome"].astype(str).str.strip()
    top = ranked.head(int(args.top_n))["genome"].tolist()
    highlight = set(top)

    win = parse_window_assignments(Path(args.window_assignments))

    fig = plt.figure(figsize=(14, 0.35 * len(top) + 6))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.35, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    draw_barcode(ax1, win, top, title=f"Top {len(top)} highest-scoring genomes (barcode)")

    ax2 = fig.add_subplot(gs[0, 1])
    draw_tree_clean(ax2, Path(args.tree_newick), highlight, title="ORF3 NJ tree (highest-scoring tips in red, *)")

    fig.tight_layout()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Done.")
    print(f"- {out_prefix.with_suffix('.pdf')}")
    print(f"- {out_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
