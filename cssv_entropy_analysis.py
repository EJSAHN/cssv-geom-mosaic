#!/usr/bin/env python3
"""
CSSV Entropy Analysis (reproducible, pipeline-style)

What it does:
1) Robustly reads .gb/.gbk/.genbank/.fa/.fasta/.fna files from --input_dir
   - tries GenBank first, then FASTA (handles "FASTA saved as .gb")
   - supports multi-record files
   - de-duplicates names by appending __dupN if needed
2) Computes circular sliding-window Shannon entropy along each genome
   - entropy computed on A/C/G/T only (N/ambiguity ignored)
3) Computes pairwise Jensen–Shannon distance (JSD) between genomes using fixed 4^k k-mer vocabulary
   - uses scipy if available; otherwise uses a safe manual JSD implementation
4) Writes CSV outputs + plots (PDF vector + PNG 300 dpi)

Outputs:
- entropy_profiles.csv
- jsd_distance_matrix.csv
- plots/entropy_landscape_heatmap.pdf + .png
- plots/jsd_heatmap.pdf + .png

Example:
python pipeline/cssv_entropy_analysis.py --input_dir "data/raw" --out_dir "results/entropy" --window 200 --step 50 --k 4
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from Bio import SeqIO

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# -----------------------------
# Robust parsing (GenBank -> FASTA fallback)
# -----------------------------
def detect_and_parse(path: Path) -> Tuple[str, List]:
    """Return (format, list[SeqRecord]). Tries GenBank then FASTA."""
    for fmt in ("genbank", "fasta"):
        try:
            recs = list(SeqIO.parse(str(path), fmt))
            if recs:
                return fmt, recs
        except Exception:
            pass
    raise ValueError(f"Could not parse {path} as GenBank or FASTA")


def clean_seq(seq: str) -> str:
    """Uppercase and keep only alphabetic characters (drop whitespace/odd chars)."""
    s = str(seq).upper()
    s = "".join([c for c in s if c.isalpha()])
    return s


def load_sequences(input_dir: Path) -> Dict[str, str]:
    exts = {".gb", ".gbk", ".genbank", ".fa", ".fasta", ".fna", ".fas", ".txt"}
    seqs: Dict[str, str] = {}
    seen: Dict[str, int] = {}

    for p in sorted(input_dir.glob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in exts:
            continue

        fmt, recs = detect_and_parse(p)
        for rec in recs:
            # match your pipeline naming convention
            name = rec.name if (fmt == "genbank" and getattr(rec, "name", None)) else rec.id
            name = str(name).strip() if name else p.stem

            # ensure unique
            if name in seen:
                seen[name] += 1
                name = f"{name}__dup{seen[name]}"
            else:
                seen[name] = 0

            seqs[name] = clean_seq(rec.seq)

    if not seqs:
        raise FileNotFoundError(
            f"No sequences parsed. Check --input_dir={input_dir} and file extensions. "
            "Also note: some .gb files may actually be FASTA; this script handles that."
        )

    return seqs


# -----------------------------
# Entropy
# -----------------------------
def shannon_entropy_acgt(seq: str) -> float:
    """
    Shannon entropy over A/C/G/T only.
    Ignores N and ambiguity codes so they don't inflate 'entropy'.
    """
    seq = seq.upper()
    counts = Counter([c for c in seq if c in "ACGT"])
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log(p, 2)
    return H


def circular_entropy_profile(seq: str, window: int, step: int) -> List[Tuple[int, float]]:
    L = len(seq)
    if L == 0:
        return []
    # circular extension
    ext = seq + seq[:window]
    out = []
    for start in range(0, L, step):
        frag = ext[start:start + window]
        H = shannon_entropy_acgt(frag)
        out.append((start, H))
    return out


# -----------------------------
# k-mer vectors + JSD
# -----------------------------
def build_vocab(k: int) -> List[str]:
    bases = ["A", "C", "G", "T"]
    voc = [""]
    for _ in range(k):
        voc = [v + b for v in voc for b in bases]
    return voc


def kmer_prob_vector(seq: str, k: int, vocab: List[str]) -> np.ndarray:
    seq = seq.upper()
    counts = Counter()
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if set(kmer) <= set("ACGT"):
            counts[kmer] += 1
            total += 1
    vec = np.zeros(len(vocab), dtype=float)
    if total == 0:
        return vec
    for j, km in enumerate(vocab):
        vec[j] = counts.get(km, 0) / total
    return vec


def jsd_distance(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    Jensen–Shannon distance (sqrt of divergence).
    Uses scipy if available; otherwise manual stable implementation.
    """
    try:
        from scipy.spatial.distance import jensenshannon  # type: ignore
        return float(jensenshannon(p, q, base=base))
    except Exception:
        # Manual: JSD(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M=0.5(P+Q)
        # Distance = sqrt(JSD)
        eps = 1e-12
        p = np.asarray(p, dtype=float) + eps
        q = np.asarray(q, dtype=float) + eps
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)

        def kl(a, b):
            return float(np.sum(a * (np.log(a / b) / np.log(base))))

        jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        return float(np.sqrt(max(jsd, 0.0)))


# -----------------------------
# Plotting
# -----------------------------
def save_pdf_png(fig: plt.Figure, pdf_path: Path, png_path: Path, dpi: int = 300) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no_plots", action="store_true", help="Write CSV outputs only")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    # CSV outputs are written even when --no_plots is used, so the output
    # directory must always exist.
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load sequences robustly
    seqs = load_sequences(input_dir)
    names = sorted(seqs.keys())
    print(f"[OK] Loaded {len(names)} sequences from {input_dir}")

    # 2) Entropy profiles
    prof_rows = []
    starts_all = None

    for name in names:
        seq = seqs[name]
        prof = circular_entropy_profile(seq, window=args.window, step=args.step)
        for start, H in prof:
            prof_rows.append({"genome": name, "start": start, "entropy": H})

    df_ent = pd.DataFrame(prof_rows)
    df_ent.to_csv(out_dir / "entropy_profiles.csv", index=False)

    if not args.no_plots:
        pivot = df_ent.pivot(index="genome", columns="start", values="entropy")
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Shannon entropy (bits; A/C/G/T only)")
        ax.set_title("Genomic entropy landscape (circular sliding windows)")
        ax.set_ylabel("Genomes")
        ncol = pivot.shape[1]
        if ncol > 1:
            xt = [0, ncol // 2, ncol - 1]
            ax.set_xticks(xt)
            ax.set_xticklabels([str(pivot.columns[i]) for i in xt])
            ax.set_xlabel("Window start position (bp)")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=6)
        save_pdf_png(fig, plot_dir / "entropy_landscape_heatmap.pdf", plot_dir / "entropy_landscape_heatmap.png", dpi=args.dpi)

    # 3) JSD distance matrix on fixed vocabulary
    vocab = build_vocab(args.k)
    vecs = {name: kmer_prob_vector(seqs[name], args.k, vocab) for name in names}

    n = len(names)
    jsd_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = jsd_distance(vecs[names[i]], vecs[names[j]], base=2.0)
            jsd_mat[i, j] = d
            jsd_mat[j, i] = d

    df_jsd = pd.DataFrame(jsd_mat, index=names, columns=names)
    df_jsd.to_csv(out_dir / "jsd_distance_matrix.csv")

    if not args.no_plots:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        im = ax.imshow(jsd_mat, aspect="auto", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Jensen–Shannon distance")
        ax.set_title(rf"Pairwise entropic distance (JSD; $k={args.k}$)")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=5)
        ax.set_yticklabels(names, fontsize=5)
        save_pdf_png(fig, plot_dir / "jsd_distance_matrix.pdf", plot_dir / "jsd_distance_matrix.png", dpi=args.dpi)

    print("[DONE] Entropy & JSD analysis complete.")
    print(f"- {out_dir / 'entropy_profiles.csv'}")
    print(f"- {out_dir / 'jsd_distance_matrix.csv'}")
    if not args.no_plots:
        print(f"- {plot_dir}")


if __name__ == "__main__":
    main()
