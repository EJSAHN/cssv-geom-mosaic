#!/usr/bin/env python3
"""Prepare accession-exact FASTA and GenBank inputs for the CSSV core panel.

The utility first searches local folders and an optional multi-FASTA panel. It
then downloads only missing records from NCBI when --download_missing is used.
Each accession is written as one record per file to avoid accidental duplicate
records from combined inputs.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

EXTS = {".fa", ".fasta", ".fna", ".fas", ".gb", ".gbk", ".genbank", ".txt"}


def read_accessions(path: Path) -> List[str]:
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        x = line.strip()
        if x and not x.startswith("#"):
            vals.append(x)
    if len(vals) != len(set(vals)):
        raise ValueError("ACCESSIONS file contains duplicate entries")
    return vals


def accession_keys(record: SeqRecord) -> List[str]:
    keys = {str(record.id).strip(), str(record.name).strip()}
    accs = record.annotations.get("accessions", []) if hasattr(record, "annotations") else []
    for a in accs:
        keys.add(str(a).strip())
    out = set()
    for key in keys:
        if not key:
            continue
        out.add(key)
        if "." in key:
            out.add(key.split(".")[0])
    return sorted(out)


def parse_any(path: Path) -> List[SeqRecord]:
    for fmt in ("genbank", "fasta"):
        try:
            recs = list(SeqIO.parse(str(path), fmt))
            if recs:
                return recs
        except Exception:
            pass
    return []


def local_record_index(paths: Iterable[Path]) -> Dict[str, SeqRecord]:
    index: Dict[str, SeqRecord] = {}
    for path in paths:
        if not path.exists() or not path.is_file() or path.suffix.lower() not in EXTS:
            continue
        for rec in parse_any(path):
            for key in accession_keys(rec):
                index.setdefault(key, rec)
    return index


def collect_local_files(source_dirs: List[Path], panel_fasta: Optional[Path]) -> List[Path]:
    """Collect local inputs with individual files taking precedence over panels.

    Combined files are excluded from source directories so they cannot create
    ambiguous duplicate lookup keys. An explicitly supplied panel FASTA is
    appended last and is therefore used only when no accession-specific local
    file has already provided that record.
    """
    files: List[Path] = []
    panel_resolved = panel_fasta.resolve() if panel_fasta and panel_fasta.exists() else None
    for d in source_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.suffix.lower() not in EXTS:
                continue
            if panel_resolved is not None and p.resolve() == panel_resolved:
                continue
            low = p.stem.lower()
            if low.startswith("combined_") or low.startswith("core48"):
                continue
            files.append(p)
    if panel_resolved is not None:
        files.append(panel_fasta)
    return files


def fetch_record(accession: str, rettype: str, email: str, api_key: Optional[str], retries: int = 4) -> SeqRecord:
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    delay = 0.12 if api_key else 0.38
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with Entrez.efetch(db="nuccore", id=accession, rettype=rettype, retmode="text") as handle:
                fmt = "genbank" if rettype.startswith("gb") else "fasta"
                rec = SeqIO.read(handle, fmt)
            time.sleep(delay)
            return rec
        except Exception as exc:  # network/transient service errors
            last_error = exc
            time.sleep(delay * attempt * 2)
    raise RuntimeError(f"NCBI download failed for {accession} ({rettype}): {last_error}")


def exact_record(index: Dict[str, SeqRecord], accession: str) -> Optional[SeqRecord]:
    return index.get(accession) or index.get(accession.split(".")[0])


def normalized_copy(rec: SeqRecord, accession: str) -> SeqRecord:
    out = rec[:]
    out.id = accession
    out.name = accession
    out.description = rec.description or accession
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage CSSV core FASTA and GenBank records by accession.")
    ap.add_argument("--accessions", required=True)
    ap.add_argument("--source_dir", action="append", default=[], help="Local source folder; may be supplied more than once")
    ap.add_argument("--panel_fasta", default=None, help="Optional combined FASTA searched after source folders")
    ap.add_argument("--out_fasta_dir", required=True)
    ap.add_argument("--out_genbank_dir", required=True)
    ap.add_argument("--email", required=True, help="Email required by NCBI Entrez")
    ap.add_argument("--download_missing", action="store_true")
    ap.add_argument("--allow_missing_genbank", action="store_true")
    args = ap.parse_args()

    accessions_path = Path(args.accessions).resolve()
    accessions = read_accessions(accessions_path)
    source_dirs = [Path(x).resolve() for x in args.source_dir]
    panel_fasta = Path(args.panel_fasta).resolve() if args.panel_fasta else None
    fasta_dir = Path(args.out_fasta_dir).resolve()
    gb_dir = Path(args.out_genbank_dir).resolve()
    fasta_dir.mkdir(parents=True, exist_ok=True)
    gb_dir.mkdir(parents=True, exist_ok=True)

    files = collect_local_files(source_dirs, panel_fasta)
    local = local_record_index(files)
    api_key = os.environ.get("NCBI_API_KEY")

    fasta_records: List[SeqRecord] = []
    gb_records: List[SeqRecord] = []
    manifest_rows: List[Tuple[str, str, str, int]] = []
    missing_fasta: List[str] = []
    missing_gb: List[str] = []

    for accession in accessions:
        fasta_path = fasta_dir / f"{accession}.fasta"
        gb_path = gb_dir / f"{accession}.gb"

        rec_local = exact_record(local, accession)
        source = "local"
        if fasta_path.exists():
            rec_fa = SeqIO.read(str(fasta_path), "fasta")
            source = "existing"
        elif rec_local is not None:
            rec_fa = normalized_copy(rec_local, accession)
            SeqIO.write(rec_fa, str(fasta_path), "fasta")
        elif args.download_missing:
            try:
                rec_fa = normalized_copy(fetch_record(accession, "fasta", args.email, api_key), accession)
                SeqIO.write(rec_fa, str(fasta_path), "fasta")
                source = "NCBI"
            except Exception as exc:
                print(f"[WARN] {exc}")
                missing_fasta.append(accession)
                rec_fa = None
        else:
            missing_fasta.append(accession)
            rec_fa = None

        if rec_fa is not None:
            fasta_records.append(rec_fa)
            manifest_rows.append((accession, source, str(fasta_path), len(rec_fa.seq)))

        if gb_path.exists():
            try:
                rec_gb = SeqIO.read(str(gb_path), "genbank")
            except Exception:
                rec_gb = None
        else:
            rec_gb = None

        # A locally parsed GenBank record retains features; FASTA records do not.
        if rec_gb is None and rec_local is not None and getattr(rec_local, "features", None):
            try:
                rec_gb = normalized_copy(rec_local, accession)
                rec_gb.annotations.setdefault("molecule_type", "DNA")
                SeqIO.write(rec_gb, str(gb_path), "genbank")
            except Exception:
                rec_gb = None

        if rec_gb is None and args.download_missing:
            try:
                rec_gb = normalized_copy(fetch_record(accession, "gbwithparts", args.email, api_key), accession)
                rec_gb.annotations.setdefault("molecule_type", "DNA")
                SeqIO.write(rec_gb, str(gb_path), "genbank")
            except Exception as exc:
                print(f"[WARN] {exc}")
                missing_gb.append(accession)
        elif rec_gb is None:
            missing_gb.append(accession)

        if rec_gb is not None:
            gb_records.append(rec_gb)

    if fasta_records:
        SeqIO.write(fasta_records, str(fasta_dir.parent / "combined_core48.fasta"), "fasta")
    if gb_records:
        SeqIO.write(gb_records, str(gb_dir.parent / "combined_core48.gb"), "genbank")

    manifest = fasta_dir.parent / "input_manifest.tsv"
    manifest.write_text(
        "accession\tsequence_source\tfasta_path\tlength\n"
        + "\n".join("\t".join(map(str, row)) for row in manifest_rows)
        + "\n",
        encoding="utf-8",
    )

    print(f"[OK] FASTA records staged: {len(fasta_records)}/{len(accessions)}")
    print(f"[OK] GenBank records staged: {len(gb_records)}/{len(accessions)}")
    print(f"[OK] Manifest: {manifest}")
    if missing_fasta:
        print("[ERR] Missing FASTA:", ", ".join(missing_fasta))
        raise SystemExit(2)
    if missing_gb:
        print("[WARN] Missing GenBank annotations:", ", ".join(missing_gb))
        if not args.allow_missing_genbank:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
