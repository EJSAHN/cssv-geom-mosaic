"""
Extract nucleotide + amino-acid FASTA sequences for predicted ORFs.

Pairs with:
  - predicted_orfs.csv (from cssv_mosaic_orf_analysis.py)
  - your original .gb/.gbk/.fasta files

Outputs (under --out_dir)
- predicted_orfs.fna : nucleotide sequences
- predicted_orfs.faa : amino acid sequences (terminal stop removed if present)
- predicted_orfs_extracted.csv : metadata + extracted lengths

Example:
python pipeline/cssv_extract_predicted_orf_seqs.py ^
  --input_dir "data/raw" ^
  --predicted_orfs "results/mosaic_orf/predicted_orfs.csv" ^
  --out_dir "results/mosaic_orf/orf_seqs"

Notes
- Coordinates assumed 0-based half-open: start0 inclusive, end0 exclusive
- wrap=True: seq[start0:] + seq[:end0]
- strand can be '+'/'-' or 1/-1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


@dataclass
class SeqEntry:
    name: str
    record_id: str
    source_file: str
    seq: str


def detect_and_parse(path: Path):
    """Return (format, list[SeqRecord]). Tries GenBank then FASTA."""
    for fmt in ("genbank", "fasta"):
        try:
            recs = list(SeqIO.parse(str(path), fmt))
            if recs:
                return fmt, recs
        except Exception:
            pass
    return None, []


def load_sequences(input_dir: Path, exts: Tuple[str, ...]) -> List[SeqEntry]:
    entries: List[SeqEntry] = []
    for p in sorted(input_dir.glob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in exts:
            continue
        fmt, recs = detect_and_parse(p)
        if not recs:
            continue
        for rec in recs:
            seq = str(rec.seq).upper()
            name = rec.name if (fmt == "genbank" and getattr(rec, "name", None)) else rec.id
            entries.append(SeqEntry(name=name, record_id=rec.id, source_file=str(p), seq=seq))

    if not entries:
        raise FileNotFoundError(f"No parseable records found in {input_dir} (extensions: {exts})")
    return entries


def build_name_index(entries: List[SeqEntry]) -> Dict[str, SeqEntry]:
    index: Dict[str, SeqEntry] = {}
    collisions: Dict[str, int] = {}

    def _add(key: str, entry: SeqEntry):
        if not key:
            return
        if key in index:
            collisions[key] = collisions.get(key, 1) + 1
            return
        index[key] = entry

    for e in entries:
        _add(e.name, e)
        _add(e.record_id, e)
        _add(Path(e.source_file).stem, e)
        if "." in e.name:
            _add(e.name.split(".", 1)[0], e)

    if collisions:
        top = sorted(collisions.items(), key=lambda x: x[1], reverse=True)[:10]
        msg = ", ".join([f"{k} (x{v})" for k, v in top])
        print(f"[WARN] Some sequence keys were duplicated; kept the first occurrence for those keys: {msg}")

    return index


def slice_circular(seq: str, start0: int, end0: int, wrapped: bool) -> str:
    if not wrapped:
        return seq[start0:end0]
    return seq[start0:] + seq[:end0]


def reverse_complement(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def translate_nt(nt: str) -> str:
    aa = str(Seq(nt).translate(to_stop=False))
    if aa.endswith("*"):
        aa = aa[:-1]
    return aa


def write_fasta(records: List[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")


def normalize_orf_table(orfs: pd.DataFrame) -> pd.DataFrame:
    orfs = orfs.copy()

    # wrap column normalization
    if "wrapped" in orfs.columns and "wrap" not in orfs.columns:
        orfs = orfs.rename(columns={"wrapped": "wrap"})
    if "wrap" not in orfs.columns:
        orfs["wrap"] = False

    # if wrap is stringy
    if orfs["wrap"].dtype == object:
        orfs["wrap"] = orfs["wrap"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)

    # orf_id (if missing)
    if "orf_id" not in orfs.columns:
        orfs = orfs.sort_values(["name", "start0", "end0"]).reset_index(drop=True)
        orfs["orf_id"] = orfs.groupby("name").cumcount() + 1
        orfs["orf_id"] = orfs["orf_id"].map(lambda x: f"orf{x}")

    required = {"name", "orf_id", "strand", "start0", "end0", "nt_len", "aa_len", "wrap"}
    missing = sorted(list(required - set(orfs.columns)))
    if missing:
        raise ValueError(f"predicted_orfs.csv missing required columns: {missing}. Found: {list(orfs.columns)}")

    # strand normalization: '+'/'-' or 1/-1
    if orfs["strand"].dtype == object:
        orfs["strand"] = orfs["strand"].map({"+": 1, "-": -1}).fillna(orfs["strand"])
    orfs["strand"] = orfs["strand"].astype(int)

    orfs["start0"] = orfs["start0"].astype(int)
    orfs["end0"] = orfs["end0"].astype(int)
    orfs["nt_len"] = orfs["nt_len"].astype(int)
    orfs["aa_len"] = orfs["aa_len"].astype(int)
    orfs["wrap"] = orfs["wrap"].astype(bool)

    return orfs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with .gb/.gbk/.fasta files (multi-record OK)")
    ap.add_argument("--predicted_orfs", required=True, help="Path to predicted_orfs.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    args = ap.parse_args()

    exts = (".gb", ".gbk", ".genbank", ".fa", ".fasta", ".fna", ".txt")

    orfs = pd.read_csv(args.predicted_orfs)
    orfs = normalize_orf_table(orfs)

    entries = load_sequences(Path(args.input_dir), exts=exts)
    idx = build_name_index(entries)

    nt_fasta: List[Tuple[str, str]] = []
    aa_fasta: List[Tuple[str, str]] = []
    meta_rows = []
    not_found = []

    for _, r in orfs.iterrows():
        name = str(r["name"])
        entry = idx.get(name) or (idx.get(name.split(".", 1)[0]) if "." in name else None)
        if entry is None:
            not_found.append(name)
            continue

        start0 = int(r["start0"])
        end0 = int(r["end0"])
        wrapped = bool(r["wrap"])
        strand = int(r["strand"])

        nt = slice_circular(entry.seq, start0, end0, wrapped=wrapped)
        if strand == -1:
            nt = reverse_complement(nt)

        aa = translate_nt(nt)

        header = f"{name}|{r['orf_id']}|strand={strand}|start0={start0}|end0={end0}|wrap={wrapped}"
        nt_fasta.append((header, nt))
        aa_fasta.append((header, aa))

        meta_rows.append(
            {
                "name": name,
                "orf_id": r["orf_id"],
                "source_file": entry.source_file,
                "record_id": entry.record_id,
                "strand": strand,
                "start0": start0,
                "end0": end0,
                "wrap": wrapped,
                "nt_len_expected": int(r["nt_len"]),
                "nt_len_extracted": len(nt),
                "aa_len_expected": int(r["aa_len"]),
                "aa_len_extracted": len(aa),
            }
        )

    if not_found:
        uniq = sorted(set(not_found))
        raise ValueError(
            "Could not find sequences for some ORF 'name' values in --input_dir. "
            "First 20 missing: " + ", ".join(uniq[:20]) + (" ..." if len(uniq) > 20 else "")
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_fasta(nt_fasta, out_dir / "predicted_orfs.fna")
    write_fasta(aa_fasta, out_dir / "predicted_orfs.faa")

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out_dir / "predicted_orfs_extracted.csv", index=False)

    bad_nt = (meta["nt_len_expected"] != meta["nt_len_extracted"]).sum()
    bad_aa = (meta["aa_len_expected"] != meta["aa_len_extracted"]).sum()
    if bad_nt or bad_aa:
        print(f"[WARN] Length mismatches: nt {bad_nt} rows, aa {bad_aa} rows. Check predicted_orfs_extracted.csv")

    print("Done.")
    print(f"- {out_dir / 'predicted_orfs.fna'}")
    print(f"- {out_dir / 'predicted_orfs.faa'}")
    print(f"- {out_dir / 'predicted_orfs_extracted.csv'}")


if __name__ == "__main__":
    main()
