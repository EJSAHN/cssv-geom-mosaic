#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
addon_01_reference_projection.py

1) Parse FASTA headers (e.g., cibv.fa) and score "header richness" (provenance strength).
2) Optionally annotate existing result CSV(s) by projecting nearest-neighbor (hit/ref) header info.

No third-party deps. Safe to upload to GitHub.
"""

import argparse
import csv
import gzip
import os
import re
import sys
from collections import Counter


ACCESSION_RE = re.compile(r"^[A-Z]{1,4}\d{5,10}(\.\d+)?$")  # e.g., AJ534983, NC_012345.1
GENUS_SPECIES_RE = re.compile(r"\b[A-Z][a-z]{2,}\s+[a-z]{2,}\b")  # e.g., "Cacao swollen"
KEYWORDS = ("country", "strain", "isolate", "host", "location", "collected", "isolation", "geo", "region")


def opn(path: str):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.endswith(".gz") \
        else open(path, "rt", encoding="utf-8", errors="replace")


def fasta_headers(fasta_path: str):
    """Yield (seq_id, full_header_without_>) from FASTA."""
    with opn(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                h = line[1:].strip()
                sid = h.split()[0] if h else ""
                if sid:
                    yield sid, h


def header_features(full_header: str):
    """Return dict of simple, robust heuristics for provenance richness."""
    toks = full_header.split()
    sid = toks[0] if toks else ""
    rest = " ".join(toks[1:]) if len(toks) > 1 else ""
    low = full_header.lower()

    extra_tokens = max(0, len(toks) - 1)
    has_delims = any(c in full_header for c in ("|", ";", "/", "[", "]", "(", ")", "{", "}"))
    has_kw = any(k in low for k in KEYWORDS)
    has_species_like = bool(GENUS_SPECIES_RE.search(full_header))
    accession_like = bool(ACCESSION_RE.match(sid))

    # Score: lightweight but useful; tuned to separate "id-only" vs "meaningful header"
    score = extra_tokens
    score += 2 if has_delims else 0
    score += 2 if has_kw else 0
    score += 2 if has_species_like else 0
    score += 1 if accession_like else 0

    if score >= 6:
        level = "strong"   # likely has provenance: accession + species/location-ish tokens
    elif score >= 2:
        level = "some"     # has at least some descriptive text
    else:
        level = "id_only"  # essentially internal ID only

    return {
        "id": sid,
        "header": full_header,
        "extra_tokens": extra_tokens,
        "accession_like": int(accession_like),
        "species_like": int(has_species_like),
        "has_keywords": int(has_kw),
        "has_delims": int(has_delims),
        "score": score,
        "level": level,
        "provenance_text": rest
    }


def normalize_candidates(x: str):
    """Generate ID candidates from a CSV value (blast-like ids, pipes, whitespace, etc.)."""
    x = (x or "").strip()
    if not x:
        return []
    cands = []
    cands.append(x)
    cands.append(x.split()[0])
    if "|" in x:
        parts = [p for p in x.split("|") if p]
        cands.extend(parts)
        cands.append(parts[-1])
        cands.append(parts[0])
    # Dedup while preserving order
    seen = set()
    out = []
    for c in cands:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def pick_column(fieldnames, preferred_names):
    fn_low = {c.lower(): c for c in fieldnames}
    for p in preferred_names:
        if p in fn_low:
            return fn_low[p]
    return None


def annotate_csv(csv_path: str, ref_map: dict, nn_col: str = None, query_col: str = None):
    with open(csv_path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise ValueError("CSV has no header row.")

        fieldnames = list(rdr.fieldnames)

        # Auto-detect columns if not provided
        if query_col is None:
            query_col = pick_column(fieldnames, [
                "query", "qseqid", "query_id", "seq_id", "sequence_id", "id", "name"
            ])
        if nn_col is None:
            nn_col = pick_column(fieldnames, [
                "nearest", "nearest_neighbor", "neighbor", "nn", "best_hit", "hit",
                "subject", "sseqid", "ref", "ref_id", "target", "top_hit"
            ])

        if nn_col is None:
            print(f"[WARN] {os.path.basename(csv_path)}: nearest/hit column not found. "
                  f"Use --nn-col to specify. Skipping annotation.")
            return None

        out_path = os.path.splitext(csv_path)[0] + ".nn_annotated.csv"
        new_cols = [
            "nn_match_id", "nn_header", "nn_level", "nn_score",
            "nn_extra_tokens", "nn_provenance_text"
        ]
        out_fields = fieldnames + [c for c in new_cols if c not in fieldnames]

        with open(out_path, "wt", encoding="utf-8", newline="") as out:
            w = csv.DictWriter(out, fieldnames=out_fields)
            w.writeheader()

            matched = 0
            total = 0
            for row in rdr:
                total += 1
                nn_val = row.get(nn_col, "")
                match_id = ""
                feat = None
                for cand in normalize_candidates(nn_val):
                    if cand in ref_map:
                        match_id = cand
                        feat = ref_map[cand]
                        break

                if feat:
                    matched += 1
                    row.update({
                        "nn_match_id": match_id,
                        "nn_header": feat["header"],
                        "nn_level": feat["level"],
                        "nn_score": feat["score"],
                        "nn_extra_tokens": feat["extra_tokens"],
                        "nn_provenance_text": feat["provenance_text"],
                    })
                else:
                    row.update({
                        "nn_match_id": "",
                        "nn_header": "",
                        "nn_level": "",
                        "nn_score": "",
                        "nn_extra_tokens": "",
                        "nn_provenance_text": "",
                    })

                w.writerow(row)

    print(f"[OK] Annotated: {csv_path} -> {out_path} (matched {matched}/{total} rows via '{nn_col}')")
    if query_col:
        print(f"     (auto-detected query column: '{query_col}')")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Summarize FASTA header richness and (optionally) project ref provenance into result CSVs."
    )
    ap.add_argument("--fasta", required=True, help="Reference FASTA (e.g., cibv.fa or .gz)")
    ap.add_argument("--csv", nargs="*", default=[], help="0+ result CSVs to annotate (optional)")
    ap.add_argument("--nn-col", default=None, help="Nearest/hit/ref column name in CSV (override autodetect)")
    ap.add_argument("--query-col", default=None, help="Query id column name in CSV (override autodetect)")
    args = ap.parse_args()

    # Load FASTA headers
    feats = []
    ref_map = {}
    dup = 0
    for sid, hdr in fasta_headers(args.fasta):
        feat = header_features(hdr)
        feats.append(feat)
        if sid in ref_map:
            dup += 1
        else:
            ref_map[sid] = feat

    if not feats:
        print("[ERROR] No FASTA headers found. Is the file valid FASTA?", file=sys.stderr)
        sys.exit(2)

    # Write header richness table
    base = os.path.splitext(os.path.basename(args.fasta))[0]
    out_tsv = os.path.join(os.path.dirname(args.fasta) or ".", f"{base}.header_richness.tsv")
    with open(out_tsv, "wt", encoding="utf-8", newline="") as out:
        out.write("\t".join([
            "id", "level", "score", "extra_tokens", "accession_like", "species_like",
            "has_keywords", "has_delims", "provenance_text", "header"
        ]) + "\n")
        for f in feats:
            out.write("\t".join([
                f["id"], f["level"], str(f["score"]), str(f["extra_tokens"]),
                str(f["accession_like"]), str(f["species_like"]),
                str(f["has_keywords"]), str(f["has_delims"]),
                f["provenance_text"].replace("\t", " "),
                f["header"].replace("\t", " "),
            ]) + "\n")

    # Print summary
    ctr = Counter(f["level"] for f in feats)
    total = len(feats)
    print(f"[OK] Parsed FASTA: {args.fasta}")
    print(f"     sequences: {total} (unique ids: {len(ref_map)}, dup ids: {dup})")
    for k in ("strong", "some", "id_only"):
        print(f"     {k:7s}: {ctr.get(k, 0)} ({ctr.get(k, 0)/total:.1%})")
    print(f"[OK] Wrote header table: {out_tsv}")

    # Optionally annotate CSVs
    for p in args.csv:
        try:
            annotate_csv(p, ref_map, nn_col=args.nn_col, query_col=args.query_col)
        except Exception as e:
            print(f"[ERROR] Failed to annotate {p}: {e}", file=sys.stderr)

    print("[DONE]")


if __name__ == "__main__":
    main()