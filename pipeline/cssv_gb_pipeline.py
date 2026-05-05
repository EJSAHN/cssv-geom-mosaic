"""
CSSV / genome analysis pipeline from local .gb/.gbk/.fasta files.

What it does (sequence-only; works even if some files are FASTA but saved with .gb extension):
1) Reads all records from the input folder (tries GenBank first, then FASTA).
2) Writes a genome_summary.csv (length, GC, N fraction, base counts, entropy).
3) Computes a simple k-mer (default 4-mer) cosine distance matrix between genomes.
4) Runs metric MDS on that distance matrix and saves genome_embedding.csv (+ an optional PDF plot).
5) (Optional) Sliding-window mosaic barcode using k-mer frequencies + KMeans.

Requirements: biopython, pandas, numpy, scikit-learn, matplotlib
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.manifold import MDS
from sklearn.cluster import KMeans


def detect_and_parse(path: Path):
    """Return (format, list[SeqRecord]). Tries GenBank then FASTA."""
    for fmt in ("genbank", "fasta"):
        try:
            recs = list(SeqIO.parse(str(path), fmt))
            if recs:
                return fmt, recs
        except Exception:
            pass
    raise ValueError(f"Could not parse {path} as GenBank or FASTA")


def shannon_entropy_base(seq: str) -> float:
    seq = seq.upper()
    counts = {b: seq.count(b) for b in "ACGT"}
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    ent = 0.0
    for b, c in counts.items():
        if c == 0:
            continue
        p = c / total
        ent -= p * math.log(p, 2)
    return ent


def kmer_freq(seq: str, k: int) -> Dict[str, float]:
    seq = seq.upper()
    valid = set("ATGC")
    counts: Counter = Counter()
    n = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if set(kmer) <= valid:
            counts[kmer] += 1
            n += 1
    if n == 0:
        return {}
    return {km: c / n for km, c in counts.items()}


def cosine_distance(vec1: Dict[str, float], vec2: Dict[str, float], keys: List[str]) -> float:
    a = np.array([vec1.get(k, 0.0) for k in keys], dtype=float)
    b = np.array([vec2.get(k, 0.0) for k in keys], dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    cos = float(np.dot(a, b) / (na * nb))
    return 1.0 - cos


def load_genomes(input_dir: Path, exts: Tuple[str, ...]) -> List[Tuple[str, str, Path, str]]:
    """
    Returns list of tuples:
      (genome_name, format_detected, source_path, sequence_string)

    If duplicate genome_name occurs (common with multi-record GB or merged files),
    we suffix it as __dupN to avoid silent overwriting later.
    """
    genomes = []
    seen = {}  # base_name -> dup_count

    for p in sorted(input_dir.glob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in exts:
            continue

        fmt, recs = detect_and_parse(p)
        for rec in recs:
            # Prefer LOCUS/name if present; otherwise fall back to record id
            genome_name = rec.name if (fmt == "genbank" and rec.name) else rec.id
            genome_name = str(genome_name).strip() if genome_name else p.stem

            # ------- make genome_name unique ----
            if genome_name in seen:
                seen[genome_name] += 1
                genome_name = f"{genome_name}__dup{seen[genome_name]}"
            else:
                seen[genome_name] = 0
            # --------------------------------------

            seq = str(rec.seq).upper()
            genomes.append((genome_name, fmt, p, seq))

    if not genomes:
        raise FileNotFoundError(f"No genome files found in {input_dir} with extensions: {exts}")
    return genomes



def write_genome_summary(genomes, out_csv: Path, default_topology: str = "unknown") -> pd.DataFrame:
    rows = []
    for name, fmt, path, seq in genomes:
        length = len(seq)
        counts = {b: seq.count(b) for b in "ACGTN"}
        n_frac = counts["N"] / length if length else float("nan")
        atgc_total = counts["A"] + counts["C"] + counts["G"] + counts["T"]
        gc = (counts["G"] + counts["C"]) / atgc_total if atgc_total else float("nan")

        topology = default_topology
        if fmt == "genbank":
            # Try to recover topology from the record by reparsing once.
            # (We keep it simple; for large datasets you can cache this.)
            rec = next(SeqIO.parse(str(path), "genbank"))
            topology = rec.annotations.get("topology", default_topology) or default_topology

        rows.append(
            {
                "name": name,
                "topology": topology,
                "source_file": str(path),
                "length": length,
                "gc": gc,
                "n_frac": n_frac,
                "entropy": shannon_entropy_base(seq),
                **{b: counts[b] for b in "ACGTN"},
            }
        )
    df = pd.DataFrame(rows).sort_values(["name"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def genome_distance_matrix(genomes, k: int, out_csv: Path) -> pd.DataFrame:
    freqs = {name: kmer_freq(seq, k=k) for name, _, _, seq in genomes}
    all_keys = sorted({km for d in freqs.values() for km in d.keys()})
    names = list(freqs.keys())

    dist = np.zeros((len(names), len(names)), dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            dist[i, j] = cosine_distance(freqs[a], freqs[b], all_keys)
    df = pd.DataFrame(dist, index=names, columns=names)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    return df


def mds_embedding(dist_df: pd.DataFrame, out_csv: Path, random_state: int = 0) -> pd.DataFrame:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)
    coords = mds.fit_transform(dist_df.values)
    emb = pd.DataFrame(coords, columns=["mds1", "mds2"])
    emb.insert(0, "name", dist_df.index.tolist())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    emb.to_csv(out_csv, index=False)
    return emb


def window_vectors(genomes, window: int, step: int, k: int) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Returns:
      windows_df: columns [name, start, end]
      X: (n_windows x n_features) dense matrix of k-mer frequencies
      feature_keys: list of kmers in order
    """
    # Collect all window kmer dicts first to get global feature space
    window_dicts = []
    meta_rows = []
    for name, _, _, seq in genomes:
        L = len(seq)
        for start in range(0, max(1, L - window + 1), step):
            end = start + window
            subseq = seq[start:end]
            d = kmer_freq(subseq, k=k)
            window_dicts.append(d)
            meta_rows.append({"name": name, "start": start, "end": end})

    feature_keys = sorted({km for d in window_dicts for km in d.keys()})
    X = np.zeros((len(window_dicts), len(feature_keys)), dtype=float)
    for i, d in enumerate(window_dicts):
        for j, km in enumerate(feature_keys):
            X[i, j] = d.get(km, 0.0)

    windows_df = pd.DataFrame(meta_rows)
    return windows_df, X, feature_keys


def cluster_windows(
    genomes,
    out_csv: Path,
    window: int = 250,
    step: int = 50,
    k: int = 4,
    n_clusters: int = 8,
    random_state: int = 0,
) -> pd.DataFrame:
    windows_df, X, _ = window_vectors(genomes, window=window, step=step, k=k)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    windows_df = windows_df.copy()
    windows_df["label"] = labels
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    windows_df.to_csv(out_csv, index=False)
    return windows_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing .gb/.gbk/.fa/.fasta files")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--k", type=int, default=4, help="k-mer size for genome distance + windows")
    ap.add_argument("--default_topology", default="unknown", help="Topology for FASTA records")
    ap.add_argument("--do_windows", action="store_true", help="Also run sliding-window clustering")
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--n_clusters", type=int, default=8)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    exts = (".gb", ".gbk", ".genbank", ".fa", ".fasta", ".fna", ".txt")

    genomes = load_genomes(input_dir, exts=exts)

    summary_csv = out_dir / "genome_summary.csv"
    dist_csv = out_dir / f"k{args.k}_cosine_distance.csv"
    emb_csv = out_dir / "genome_embedding.csv"
    windows_csv = out_dir / "window_assignments.csv"

    write_genome_summary(genomes, summary_csv, default_topology=args.default_topology)
    dist_df = genome_distance_matrix(genomes, k=args.k, out_csv=dist_csv)
    mds_embedding(dist_df, out_csv=emb_csv)

    if args.do_windows:
        cluster_windows(
            genomes,
            out_csv=windows_csv,
            window=args.window,
            step=args.step,
            k=args.k,
            n_clusters=args.n_clusters,
        )

    print("Done.")
    print(f"- {summary_csv}")
    print(f"- {dist_csv}")
    print(f"- {emb_csv}")
    if args.do_windows:
        print(f"- {windows_csv}")


if __name__ == "__main__":
    main()
