"""
CSSV mosaic (window cluster labels) <-> ORF boundary proximity analysis.

Inputs
- --input_dir: folder containing .gb/.gbk/.fa/.fasta (can be GenBank or FASTA)
- --window_assignments: CSV from cssv_gb_pipeline.py (columns: name,start,end,label)
Outputs (written to --out_dir)
- predicted_orfs.csv
- switchpoints.csv
- switchpoint_orf_distances.csv
- enrichment_summary.csv
- plots/ (optional): distance plots + top genome tracks

Notes
- ORFs are predicted from sequence only (6-frame). No need for CDS features in GenBank.
- Coordinates: 0-based, end-exclusive for internal computation; 1-based fields are also saved for readability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO

import matplotlib.pyplot as plt


# ----------------------------
# Parsing
# ----------------------------
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


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1]


def load_sequences_by_name(input_dir: Path, exts: Tuple[str, ...]) -> Dict[str, str]:
    """
    Matches the naming convention used in your cssv_gb_pipeline.py:
      genome_name = rec.name if (fmt == "genbank" and rec.name) else rec.id

    Returns dict: {name: sequence}
    If duplicate names occur, raises a ValueError (because window_assignments would be ambiguous too).
    """
    seqs: Dict[str, str] = {}
    duplicates: List[Tuple[str, str]] = []

    for p in sorted(input_dir.glob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in exts:
            continue

        fmt, recs = detect_and_parse(p)
        for rec in recs:
            name = rec.name if (fmt == "genbank" and getattr(rec, "name", None)) else rec.id
            name = (name or "").strip()
            if not name:
                # fallback: filename stem
                name = p.stem

            seq = str(rec.seq).upper()

            if name in seqs and seqs[name] != seq:
                duplicates.append((name, str(p)))
            else:
                seqs[name] = seq

    if duplicates:
        msg = ["Duplicate genome names detected (ambiguous with window_assignments.csv).",
               "Fix: patch cssv_gb_pipeline.py to disambiguate names (seen=set()) and rerun windows.",
               "Duplicates:"]
        msg += [f" - name={n}, file={fp}" for n, fp in duplicates[:30]]
        raise ValueError("\n".join(msg))

    if not seqs:
        raise FileNotFoundError(f"No sequence files found in {input_dir}")

    return seqs


# ----------------------------
# ORF prediction (6-frame)
# ----------------------------
STOP_CODONS = {"TAA", "TAG", "TGA"}


@dataclass
class ORF:
    name: str
    strand: str         # '+' or '-'
    frame: int          # 0,1,2
    start0: int         # 0-based start (boundary)
    end0: int           # 0-based end-exclusive (boundary), mod L for circular may be 0..L-1
    span_start0: int    # region start on the 0..L axis for plotting (linearized)
    span_end0: int      # region end on the 0..L axis for plotting (linearized)
    wrap: bool          # True if span crosses origin (circular only)
    nt_len: int
    aa_len: int
    start_codon: str
    stop_codon: str


def scan_orfs_one_strand(
    s: str,
    L: int,
    name: str,
    strand: str,
    circular: bool,
    start_codons: List[str],
    min_aa: int,
    allow_nested: bool,
) -> List[Tuple[int, int, int, str, str]]:
    """
    Returns list of tuples:
      (start_nt_in_scan, end_nt_excl_in_scan, frame, start_codon, stop_codon)
    Coordinates are in the scan string (which may be doubled if circular).
    """
    scan = s + (s if circular else "")
    scan_len = len(scan)

    # start positions must be in first L nts to avoid duplicates on circular
    start_limit = L if circular else L  # same, but kept for clarity
    results: List[Tuple[int, int, int, str, str]] = []

    for frame in (0, 1, 2):
        i = frame
        while i + 3 <= start_limit:
            codon = scan[i:i+3]
            if codon in start_codons:
                # find first in-frame stop codon after i
                j = i + 3
                found = False
                while j + 3 <= scan_len and (j - i) <= L:
                    codon2 = scan[j:j+3]
                    if codon2 in STOP_CODONS:
                        end = j + 3
                        nt_len = end - i
                        aa_len = (nt_len // 3) - 1  # exclude stop codon
                        if aa_len >= min_aa:
                            results.append((i, end, frame, codon, codon2))
                        found = True
                        if allow_nested:
                            i += 3
                        else:
                            i = end  # jump past the stop codon (non-overlapping in this frame)
                        break
                    j += 3
                if not found:
                    i += 3
            else:
                i += 3

    return results


def predict_orfs_for_genome(
    name: str,
    seq: str,
    circular: bool,
    start_codons: List[str],
    min_aa: int,
    allow_nested: bool,
) -> List[ORF]:
    L = len(seq)
    if L == 0:
        return []

    orfs: List[ORF] = []

    # + strand scan
    plus_hits = scan_orfs_one_strand(
        s=seq, L=L, name=name, strand="+",
        circular=circular, start_codons=start_codons, min_aa=min_aa, allow_nested=allow_nested
    )
    for start_scan, end_scan, frame, sc, stc in plus_hits:
        start0 = start_scan % L
        end0 = end_scan % L
        nt_len = end_scan - start_scan
        aa_len = (nt_len // 3) - 1

        if circular:
            span_start0 = start0
            span_end0 = end0
            wrap = (end_scan > L) or (span_start0 >= span_end0)
        else:
            span_start0 = min(start0, end0) if start0 != end0 else start0
            span_end0 = max(start0, end0) if start0 != end0 else min(L, start0 + nt_len)
            wrap = False

        orfs.append(ORF(
            name=name, strand="+", frame=frame,
            start0=start0, end0=end0,
            span_start0=span_start0, span_end0=span_end0, wrap=wrap,
            nt_len=nt_len, aa_len=aa_len, start_codon=sc, stop_codon=stc
        ))

    # - strand scan (scan reverse complement, then map back)
    rc = reverse_complement(seq)
    minus_hits = scan_orfs_one_strand(
        s=rc, L=L, name=name, strand="-",
        circular=circular, start_codons=start_codons, min_aa=min_aa, allow_nested=allow_nested
    )
    for start_scan, end_scan, frame, sc, stc in minus_hits:
        # map scan coords on rc back to original boundaries (0..L)
        # boundaries in original:
        #   start boundary = L - end_rc
        #   end boundary   = L - start_rc
        start0 = (L - (end_scan % L)) % L
        end0 = (L - (start_scan % L)) % L

        nt_len = end_scan - start_scan
        aa_len = (nt_len // 3) - 1

        if circular:
            # For minus strand, the span on the linearized 0..L axis is from end0 -> start0 going forward
            span_start0 = end0
            span_end0 = start0
            wrap = (end_scan > L) or (span_start0 >= span_end0)
        else:
            span_start0 = min(start0, end0)
            span_end0 = max(start0, end0)
            wrap = False

        orfs.append(ORF(
            name=name, strand="-", frame=frame,
            start0=start0, end0=end0,
            span_start0=span_start0, span_end0=span_end0, wrap=wrap,
            nt_len=nt_len, aa_len=aa_len, start_codon=sc, stop_codon=stc
        ))

    # sort: long ORFs first
    orfs.sort(key=lambda x: (-x.aa_len, x.name, x.strand, x.start0))
    return orfs


def orfs_to_dataframe(orfs: List[ORF]) -> pd.DataFrame:
    rows = []
    for o in orfs:
        rows.append({
            "name": o.name,
            "strand": o.strand,
            "frame": o.frame,
            # 0-based end-exclusive boundaries
            "start0": int(o.start0),
            "end0": int(o.end0),
            "span_start0": int(o.span_start0),
            "span_end0": int(o.span_end0),
            "wrap": bool(o.wrap),
            "nt_len": int(o.nt_len),
            "aa_len": int(o.aa_len),
            "start_codon": o.start_codon,
            "stop_codon": o.stop_codon,
            # also write 1-based inclusive-ish fields for human reading
            "start1": int(o.start0) + 1,
            "end1": int(o.end0) if o.end0 != 0 else int(o.end0),  # for circular end0==0 means wraps
        })
    return pd.DataFrame(rows)


# ----------------------------
# Switchpoints from windows
# ----------------------------
def _run_length_encode(labels: list[int]) -> list[tuple[int, int, int]]:
    """Return (start_index, end_index, label) runs for a linear label list."""
    if not labels:
        return []
    runs = []
    run_start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            runs.append((run_start, i - 1, int(labels[i - 1])))
            run_start = i
    runs.append((run_start, len(labels) - 1, int(labels[-1])))
    return runs


def extract_switchpoints(
    windows_df: pd.DataFrame,
    switchpoint_mode: str = "start",  # 'start' or 'mid'
    min_run: int = 1,
    circular: bool = False,
) -> pd.DataFrame:
    """Extract label-transition positions from ordered barcode windows.

    For circular tracks, the last and first windows are treated as adjacent.
    The label sequence is rotated to a genuine transition before run-length
    encoding, which prevents the same circular run from being split at the
    arbitrary coordinate origin.
    """
    need = {"name", "start", "end", "label"}
    if not need.issubset(set(windows_df.columns)):
        raise ValueError(
            f"window_assignments must have columns {sorted(need)}. "
            f"Got {windows_df.columns.tolist()}"
        )

    out_rows = []
    for name, g in windows_df.groupby("name", sort=False):
        gg = g.sort_values("start").reset_index(drop=True)
        labels0 = gg["label"].astype(int).tolist()
        starts0 = gg["start"].astype(int).tolist()
        ends0 = gg["end"].astype(int).tolist()
        n = len(gg)
        if n <= 1 or len(set(labels0)) <= 1:
            continue

        # Rotate a circular sequence to begin immediately after a transition.
        # This ensures the first and last runs are distinct and each circular
        # run is represented exactly once.
        offset = 0
        if circular:
            for i in range(n):
                if labels0[i] != labels0[(i - 1) % n]:
                    offset = i
                    break

        order = [(offset + i) % n for i in range(n)]
        labels = [labels0[i] for i in order]
        starts = [starts0[i] for i in order]
        ends = [ends0[i] for i in order]
        original_indices = order
        runs = _run_length_encode(labels)

        transition_count = len(runs) if circular else len(runs) - 1
        for r in range(transition_count):
            nr = (r + 1) % len(runs)
            a0, a1, la = runs[r]
            b0, b1, lb = runs[nr]
            len_a = a1 - a0 + 1
            len_b = b1 - b0 + 1
            if len_a < min_run or len_b < min_run:
                continue

            if switchpoint_mode == "start":
                pos0 = int(starts[b0])
            elif switchpoint_mode == "mid":
                prev_end = int(ends[a1])
                new_start = int(starts[b0])
                if circular:
                    if "genome_len" in gg.columns:
                        L = int(gg["genome_len"].iloc[0])
                    else:
                        L = max(starts0) + 1
                    prev_mod = prev_end % L
                    delta = (new_start - prev_mod) % L
                    pos0 = int((prev_mod + delta / 2.0) % L)
                else:
                    pos0 = int((prev_end + new_start) // 2)
            else:
                raise ValueError("--switchpoint_mode must be 'start' or 'mid'")

            out_rows.append(
                {
                    "name": name,
                    "pos0": pos0,
                    "prev_label": int(la),
                    "new_label": int(lb),
                    "prev_run_windows": int(len_a),
                    "new_run_windows": int(len_b),
                    "circular_transition": bool(circular),
                    "new_run_start_window_index": int(original_indices[b0]),
                }
            )

    columns = [
        "name", "pos0", "prev_label", "new_label",
        "prev_run_windows", "new_run_windows",
        "circular_transition", "new_run_start_window_index",
    ]
    return pd.DataFrame(out_rows, columns=columns)


# ----------------------------
# Distance + permutation
# ----------------------------
def circular_distance(a: int, b: int, L: int) -> int:
    d = abs(a - b)
    return int(min(d, L - d))


def nearest_boundary_distance(
    pos0: int,
    boundaries0: np.ndarray,
    L: int,
    circular: bool
) -> float:
    if boundaries0.size == 0:
        return float("nan")
    if circular:
        dists = np.array([circular_distance(int(pos0), int(b), L) for b in boundaries0], dtype=float)
    else:
        dists = np.abs(boundaries0.astype(float) - float(pos0))
    return float(np.min(dists))


def compute_distances(
    seqs: Dict[str, str],
    orf_df: pd.DataFrame,
    switch_df: pd.DataFrame,
    circular: bool
) -> pd.DataFrame:
    # boundaries per genome
    boundaries: Dict[str, np.ndarray] = {}
    for name, g in orf_df.groupby("name", sort=False):
        # ORF boundary positions: start0 and end0 (both matter)
        bs = pd.concat([g["start0"], g["end0"]], ignore_index=True).dropna().astype(int).values
        # keep within [0, L-1] in circular sense (end0 can be 0, OK)
        boundaries[name] = bs

    rows = []
    for _, r in switch_df.iterrows():
        name = r["name"]
        pos0 = int(r["pos0"])
        if name not in seqs:
            continue
        L = len(seqs[name])
        b = boundaries.get(name, np.array([], dtype=int))
        d = nearest_boundary_distance(pos0=pos0, boundaries0=b, L=L, circular=circular)

        rows.append({
            "name": name,
            "pos0": pos0,
            "pos1": pos0 + 1,
            "prev_label": int(r["prev_label"]),
            "new_label": int(r["new_label"]),
            "dist_to_nearest_orf_boundary_bp": d,
            "genome_len": L,
        })
    return pd.DataFrame(rows)


def permutation_enrichment(
    seqs: Dict[str, str],
    boundaries_by_name: Dict[str, np.ndarray],
    switch_counts_by_name: Dict[str, int],
    near_bp: int,
    perm: int,
    circular: bool,
    seed: int = 0
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    # observed
    obs_total = 0
    obs_near = 0
    for name, n_switch in switch_counts_by_name.items():
        if n_switch <= 0:
            continue
        L = len(seqs[name])
        # observed switchpoint positions are handled elsewhere; here we only compute null
        # we'll compute observed later from distance table
        obs_total += n_switch

    # null distribution of "fraction near boundaries" (global)
    null_fracs = np.zeros(perm, dtype=float)
    total_switch = sum(max(0, c) for c in switch_counts_by_name.values())
    if total_switch == 0:
        return {
            "total_switchpoints": 0,
            "near_bp": near_bp,
            "perm": perm,
            "null_mean_frac": float("nan"),
            "null_std_frac": float("nan"),
            "empirical_p_one_sided": float("nan"),
        }

    for t in range(perm):
        near_count = 0
        for name, n_switch in switch_counts_by_name.items():
            if n_switch <= 0:
                continue
            L = len(seqs[name])
            b = boundaries_by_name.get(name, np.array([], dtype=int))
            if b.size == 0:
                continue
            rand_pos = rng.integers(low=0, high=L, size=n_switch, endpoint=False)
            # compute nearest boundary distance for each rand_pos
            if circular:
                # vectorized-ish via loop (L small, n_switch moderate)
                for p in rand_pos:
                    d = nearest_boundary_distance(int(p), b, L=L, circular=True)
                    if d <= near_bp:
                        near_count += 1
            else:
                # linear: distances can be vectorized
                for p in rand_pos:
                    d = nearest_boundary_distance(int(p), b, L=L, circular=False)
                    if d <= near_bp:
                        near_count += 1

        null_fracs[t] = near_count / total_switch

    return {
        "total_switchpoints": int(total_switch),
        "near_bp": int(near_bp),
        "perm": int(perm),
        "null_mean_frac": float(np.mean(null_fracs)),
        "null_std_frac": float(np.std(null_fracs, ddof=1)) if perm > 1 else float("nan"),
        "null_fracs": null_fracs,  # caller may save as npy or summarize
    }


# ----------------------------
# Plotting
# ----------------------------
def plot_distance_hist(observed_dist: np.ndarray, out_pdf: Path, out_png: Path):
    observed_dist = observed_dist[np.isfinite(observed_dist)]
    plt.figure()
    plt.hist(observed_dist, bins=50)
    plt.xlabel("Distance to nearest ORF boundary (bp)")
    plt.ylabel("Switchpoint count")
    plt.title("Switchpoint → ORF boundary distance (observed)")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_top_genome_track(
    name: str,
    seq: str,
    windows_df: pd.DataFrame,
    orf_df: pd.DataFrame,
    switch_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    max_orfs: int = 30,
):
    L = len(seq)
    gwin = windows_df[windows_df["name"] == name].sort_values("start")
    gorfs = orf_df[orf_df["name"] == name].copy()
    gsw = switch_df[switch_df["name"] == name].copy()

    plt.figure(figsize=(12, 4))
    ax = plt.gca()

    # Mosaic runs (use run-length encoding for cleaner bars)
    if len(gwin) > 0:
        starts = gwin["start"].astype(int).to_list()
        ends = gwin["end"].astype(int).to_list()
        labels = gwin["label"].astype(int).to_list()

        runs = []
        rs = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                runs.append((rs, i-1, labels[i-1]))
                rs = i
        runs.append((rs, len(labels)-1, labels[-1]))

        cmap = plt.get_cmap("tab20")
        for a0, a1, lab in runs:
            x0 = max(0, starts[a0])
            x1 = min(L, ends[a1])
            ax.broken_barh([(x0, max(1, x1 - x0))], (3.0, 0.8), facecolors=cmap(lab % 20))

        ax.text(0, 3.95, "Mosaic labels", fontsize=9, va="top")

    # ORFs (top N by aa_len)
    if len(gorfs) > 0:
        gorfs = gorfs.sort_values("aa_len", ascending=False).head(max_orfs)
        for _, r in gorfs.iterrows():
            span_start = int(r["span_start0"])
            span_end = int(r["span_end0"])
            wrap = bool(r["wrap"])
            strand = r["strand"]

            if not wrap:
                ax.broken_barh([(span_start, max(1, span_end - span_start))], (1.8, 0.8))
            else:
                ax.broken_barh([(span_start, max(1, L - span_start))], (1.8, 0.8))
                ax.broken_barh([(0, max(1, span_end))], (1.8, 0.8))

        ax.text(0, 2.75, f"Predicted ORFs (top {max_orfs})", fontsize=9, va="top")

    # Switchpoints
    if len(gsw) > 0:
        xs = gsw["pos0"].astype(int).to_numpy()
        ys = np.full_like(xs, 1.3, dtype=float)
        ax.scatter(xs, ys, s=10)
        ax.text(0, 1.55, "Switchpoints", fontsize=9, va="bottom")

    ax.set_xlim(0, L)
    ax.set_ylim(0.8, 4.2)
    ax.set_yticks([])
    ax.set_xlabel("Genome position (bp)")
    ax.set_title(name)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing .gb/.gbk/.fa/.fasta files")
    ap.add_argument("--window_assignments", required=True, help="CSV from cssv_gb_pipeline.py")
    ap.add_argument("--out_dir", required=True, help="Output folder")

    ap.add_argument("--circular", action="store_true", help="Treat genomes as circular for ORF wrap + circular distance")
    ap.add_argument("--start_codons", default="ATG", help="Comma-separated start codons (default ATG). e.g. ATG,GTG,TTG")
    ap.add_argument("--min_orf_aa", type=int, default=300, help="Minimum ORF protein length (aa), excluding stop codon")
    ap.add_argument("--allow_nested", action="store_true", help="Allow nested/overlapping ORFs (more hits)")

    ap.add_argument("--switchpoint_mode", default="start", choices=["start", "mid"], help="How to place switchpoint position")
    ap.add_argument("--min_run", type=int, default=1, help="Minimum windows per run on BOTH sides to count a switchpoint")

    ap.add_argument("--near_bp", type=int, default=200, help="Distance threshold for 'near boundary' enrichment")
    ap.add_argument("--perm", type=int, default=2000, help="Permutation count for null distribution")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")

    ap.add_argument("--plot_top", type=int, default=10, help="Plot top N genomes by number of switchpoints (0 to disable)")
    ap.add_argument("--max_orfs_plot", type=int, default=30, help="Max ORFs to draw per genome plot")
    ap.add_argument("--no_plots", action="store_true", help="Write tabular outputs only")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = (".gb", ".gbk", ".genbank", ".fa", ".fasta", ".fna", ".txt")
    start_codons = [x.strip().upper() for x in args.start_codons.split(",") if x.strip()]
    if not start_codons:
        start_codons = ["ATG"]

    # 1) Load sequences
    seqs = load_sequences_by_name(input_dir, exts=exts)

    # 2) Read windows + extract switchpoints
    windows_df = pd.read_csv(args.window_assignments)
    switch_df = extract_switchpoints(
        windows_df,
        switchpoint_mode=args.switchpoint_mode,
        min_run=args.min_run,
        circular=args.circular,
    )
    switch_csv = out_dir / "switchpoints.csv"
    switch_df.to_csv(switch_csv, index=False)

    # 3) Predict ORFs
    all_orfs: List[ORF] = []
    for name, seq in seqs.items():
        all_orfs.extend(
            predict_orfs_for_genome(
                name=name,
                seq=seq,
                circular=args.circular,
                start_codons=start_codons,
                min_aa=args.min_orf_aa,
                allow_nested=args.allow_nested,
            )
        )

    orf_df = orfs_to_dataframe(all_orfs)
    orf_csv = out_dir / "predicted_orfs.csv"
    orf_df.to_csv(orf_csv, index=False)

    # 4) Distances
    dist_df = compute_distances(seqs=seqs, orf_df=orf_df, switch_df=switch_df, circular=args.circular)
    dist_csv = out_dir / "switchpoint_orf_distances.csv"
    dist_df.to_csv(dist_csv, index=False)

    # 5) Enrichment (global)
    # boundaries per genome
    boundaries_by_name: Dict[str, np.ndarray] = {}
    for name, g in orf_df.groupby("name", sort=False):
        bs = pd.concat([g["start0"], g["end0"]], ignore_index=True).dropna().astype(int).values
        boundaries_by_name[name] = bs

    switch_counts = switch_df.groupby("name").size().to_dict()

    null = permutation_enrichment(
        seqs=seqs,
        boundaries_by_name=boundaries_by_name,
        switch_counts_by_name=switch_counts,
        near_bp=args.near_bp,
        perm=args.perm,
        circular=args.circular,
        seed=args.seed,
    )

    # observed fraction within near_bp
    obs = dist_df["dist_to_nearest_orf_boundary_bp"].to_numpy()
    obs = obs[np.isfinite(obs)]
    obs_frac = float(np.mean(obs <= args.near_bp)) if obs.size else float("nan")

    null_fracs = null.pop("null_fracs")
    p_one_sided = float((1 + np.sum(null_fracs >= obs_frac)) / (len(null_fracs) + 1)) if np.isfinite(obs_frac) else float("nan")
    z = float((obs_frac - np.mean(null_fracs)) / (np.std(null_fracs, ddof=1) + 1e-12)) if np.isfinite(obs_frac) else float("nan")

    summary = {
        **null,
        "observed_frac_near": obs_frac,
        "empirical_p_one_sided": p_one_sided,
        "z_score": z,
        "circular": bool(args.circular),
        "min_orf_aa": int(args.min_orf_aa),
        "start_codons": start_codons,
        "switchpoint_mode": args.switchpoint_mode,
        "min_run": int(args.min_run),
    }

    # write summary
    summary_csv = out_dir / "enrichment_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    # also save null fracs (so you can replot without recomputing)
    npy_path = out_dir / "null_fracs.npy"
    np.save(npy_path, null_fracs)

    # 6) Optional plots
    plots_dir = out_dir / "plots"
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_distance_hist(
            observed_dist=dist_df["dist_to_nearest_orf_boundary_bp"].to_numpy(),
            out_pdf=plots_dir / "distance_hist_observed.pdf",
            out_png=plots_dir / "distance_hist_observed.png",
        )

        if args.plot_top and args.plot_top > 0:
            counts = switch_df.groupby("name").size().sort_values(ascending=False)
            top_names = counts.head(args.plot_top).index.tolist()
            for nm in top_names:
                if nm not in seqs:
                    continue
                plot_top_genome_track(
                    name=nm,
                    seq=seqs[nm],
                    windows_df=windows_df,
                    orf_df=orf_df,
                    switch_df=switch_df,
                    out_pdf=plots_dir / f"{nm}_track.pdf",
                    out_png=plots_dir / f"{nm}_track.png",
                    max_orfs=args.max_orfs_plot,
                )

    print("Done.")
    print(f"- ORFs: {orf_csv}")
    print(f"- Switchpoints: {switch_csv}")
    print(f"- Distances: {dist_csv}")
    print(f"- Summary: {summary_csv}")
    print(f"- Null fracs: {npy_path}")
    if not args.no_plots:
        print(f"- Plots: {plots_dir}")


if __name__ == "__main__":
    main()
