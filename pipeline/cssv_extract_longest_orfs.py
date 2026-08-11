"""
Extract the longest ORF per genome from predicted_orfs.csv and (optionally) pick the matching
AA/NT sequences from predicted_orfs.faa / predicted_orfs.fna.

Works with headers like:
>AJ534983.1|orf1|strand=1|start0=1272|end0=6792|wrap=False

Inputs:
- --predicted_orfs_csv: CSV from cssv_mosaic_orf_analysis.py (predicted_orfs.csv)
Optional:
- --orf_faa: FASTA amino-acid sequences (predicted_orfs.faa)
- --orf_fna: FASTA nucleotide sequences (predicted_orfs.fna)

Outputs (in --out_dir):
- longest_orfs.csv
- longest_orfs.faa (if --orf_faa given)
- longest_orfs.fna (if --orf_fna given)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterator, Tuple, Optional

import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_strand(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if s in {"+", "+1", "1"}:
        return 1
    if s in {"-", "-1"}:
        return -1
    try:
        return int(float(s))
    except Exception:
        return None


def normalize_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def fasta_iter(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (header_without_>, sequence) for a FASTA file."""
    header = None
    seq_chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)


def parse_orf_header(header: str):
    """
    Parse headers like:
      AJ534983.1|orf1|strand=1|start0=1272|end0=6792|wrap=False
    Returns: (genome_name, orf_label, meta_dict)
    """
    parts = header.split("|")
    genome = parts[0].strip()
    orf_label = parts[1].strip() if len(parts) >= 2 else None

    meta = {}
    for p in parts[2:]:
        if "=" in p:
            k, v = p.split("=", 1)
            meta[k.strip()] = v.strip()

    # Normalize types
    if "strand" in meta:
        meta["strand"] = normalize_strand(meta["strand"])
    if "start0" in meta:
        try:
            meta["start0"] = int(meta["start0"])
        except Exception:
            pass
    if "end0" in meta:
        try:
            meta["end0"] = int(meta["end0"])
        except Exception:
            pass
    if "wrap" in meta:
        meta["wrap"] = normalize_bool(meta["wrap"])

    return genome, orf_label, meta


def write_fasta(records: list[Tuple[str, str]], out_path: Path, line_width: int = 60) -> None:
    with open(out_path, "w", encoding="utf-8") as w:
        for header, seq in records:
            w.write(f">{header}\n")
            for i in range(0, len(seq), line_width):
                w.write(seq[i : i + line_width] + "\n")


def build_seq_map(fasta_path: Path) -> Dict[Tuple[str, int, int, int, bool], Tuple[str, str]]:
    """
    Build dict:
      key = (genome, strand, start0, end0, wrap)
      val = (original_header, sequence)
    """
    m: Dict[Tuple[str, int, int, int, bool], Tuple[str, str]] = {}
    for header, seq in fasta_iter(fasta_path):
        genome, _orf_label, meta = parse_orf_header(header)
        key = (
            genome,
            int(meta.get("strand")) if meta.get("strand") is not None else None,
            int(meta.get("start0")) if meta.get("start0") is not None else None,
            int(meta.get("end0")) if meta.get("end0") is not None else None,
            bool(meta.get("wrap", False)),
        )
        m[key] = (header, seq)
    return m


def select_top_orfs(df: pd.DataFrame, top_n: int, min_aa: int) -> pd.DataFrame:
    req_cols = {"name", "start0", "end0", "aa_len"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"predicted_orfs_csv is missing required columns: {sorted(missing)}")

    out = df.copy()

    # normalize strand/wrap to match fasta header style
    if "strand" in out.columns:
        out["_strand"] = out["strand"].apply(normalize_strand)
    else:
        out["_strand"] = None

    if "wrap" in out.columns:
        out["_wrap"] = out["wrap"].apply(normalize_bool)
    else:
        out["_wrap"] = False

    out["aa_len"] = pd.to_numeric(out["aa_len"], errors="coerce")
    out["start0"] = pd.to_numeric(out["start0"], errors="coerce")
    out["end0"] = pd.to_numeric(out["end0"], errors="coerce")

    out = out.dropna(subset=["aa_len", "start0", "end0"])
    out = out[out["aa_len"] >= min_aa].copy()

    if out.empty:
        return out

    # Sorting for tie-breaks: aa_len desc, nt_len desc (if exists), start0 asc
    sort_cols = ["aa_len"]
    ascending = [False]

    if "nt_len" in out.columns:
        out["nt_len"] = pd.to_numeric(out["nt_len"], errors="coerce")
        sort_cols.append("nt_len")
        ascending.append(False)

    sort_cols.append("start0")
    ascending.append(True)

    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    picked = []
    for name, g in out.groupby("name", sort=False):
        gg = g.head(top_n).copy()
        gg["rank"] = range(1, len(gg) + 1)
        picked.append(gg)

    return pd.concat(picked, ignore_index=True) if picked else out.iloc[0:0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicted_orfs_csv", required=True, help="Path to predicted_orfs.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--orf_faa", default=None, help="Optional: predicted_orfs.faa (AA sequences)")
    ap.add_argument("--orf_fna", default=None, help="Optional: predicted_orfs.fna (NT sequences)")
    ap.add_argument("--top_n", type=int, default=1, help="Top N ORFs per genome (default: 1 = longest only)")
    ap.add_argument("--min_aa", type=int, default=0, help="Minimum ORF length (aa) to consider")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    df = pd.read_csv(args.predicted_orfs_csv)
    sel = select_top_orfs(df, top_n=max(1, args.top_n), min_aa=max(0, args.min_aa))

    if sel.empty:
        raise RuntimeError("No ORFs selected (check --min_aa or your input CSV).")

    out_csv = out_dir / "longest_orfs.csv"
    sel.to_csv(out_csv, index=False)

    # Load FASTA maps if provided
    aa_map = build_seq_map(Path(args.orf_faa)) if args.orf_faa else None
    nt_map = build_seq_map(Path(args.orf_fna)) if args.orf_fna else None

    # Write AA FASTA
    if aa_map is not None:
        aa_records = []
        miss = 0
        for _, r in sel.iterrows():
            key = (
                str(r["name"]),
                int(r["_strand"]) if r["_strand"] is not None else None,
                int(r["start0"]),
                int(r["end0"]),
                bool(r["_wrap"]),
            )
            if key not in aa_map:
                miss += 1
                continue

            _orig_header, seq = aa_map[key]
            new_header = (
                f'{r["name"]}|top{int(r["rank"])}'
                f'|strand={int(r["_strand"]) if r["_strand"] is not None else "NA"}'
                f'|start0={int(r["start0"])}|end0={int(r["end0"])}|wrap={bool(r["_wrap"])}'
                f'|aa_len={int(r["aa_len"])}'
            )
            aa_records.append((new_header, seq))

        out_faa = out_dir / "longest_orfs.faa"
        write_fasta(aa_records, out_faa)

        if miss > 0:
            print(f"[WARN] AA FASTA: {miss} selected ORFs were not found by (name,strand,start0,end0,wrap) key.", file=sys.stderr)

    # Write NT FASTA
    if nt_map is not None:
        nt_records = []
        miss = 0
        for _, r in sel.iterrows():
            key = (
                str(r["name"]),
                int(r["_strand"]) if r["_strand"] is not None else None,
                int(r["start0"]),
                int(r["end0"]),
                bool(r["_wrap"]),
            )
            if key not in nt_map:
                miss += 1
                continue

            _orig_header, seq = nt_map[key]
            new_header = (
                f'{r["name"]}|top{int(r["rank"])}'
                f'|strand={int(r["_strand"]) if r["_strand"] is not None else "NA"}'
                f'|start0={int(r["start0"])}|end0={int(r["end0"])}|wrap={bool(r["_wrap"])}'
                f'|nt_len={int(r["nt_len"]) if "nt_len" in r and pd.notna(r["nt_len"]) else "NA"}'
                f'|aa_len={int(r["aa_len"])}'
            )
            nt_records.append((new_header, seq))

        out_fna = out_dir / "longest_orfs.fna"
        write_fasta(nt_records, out_fna)

        if miss > 0:
            print(f"[WARN] NT FASTA: {miss} selected ORFs were not found by (name,strand,start0,end0,wrap) key.", file=sys.stderr)

    print("Done.")
    print(f"- {out_csv}")
    if aa_map is not None:
        print(f"- {out_dir / 'longest_orfs.faa'}")
    if nt_map is not None:
        print(f"- {out_dir / 'longest_orfs.fna'}")


if __name__ == "__main__":
    main()
