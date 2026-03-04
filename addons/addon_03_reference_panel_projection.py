#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
addon_03_reference_panel_projection.py

Purpose
-------
Project a set of "query" genomes onto an external "reference panel" using an
alignment-free k-mer cosine distance, without mixing datasets in the core pipeline.

This script computes:
  - Query -> nearest reference (minimum cosine distance)
  - Distance value
  - Reference provenance text (from FASTA header) + conservative country/isolate tags
  - Token frequency summary across nearest-reference hits

Inputs
------
  --query_fasta   FASTA containing query genomes (e.g., your 48-core dataset)
  --ref_fasta     FASTA containing reference panel genomes (e.g., PMPP 86/87)
  --k             k-mer size (default 4)

Outputs
-------
  --out           TSV mapping table (default: projection_nearest_reference.tsv)
  --summary       TSV token summary (default: projection_token_summary.tsv)
  --ref_headers   Optional: write reference header richness table (like addon_01)
  --query_headers Optional: write query header richness table (like addon_01)

Notes
-----
- No third-party dependencies (pure Python).
- Handles circularity implicitly (whole-genome k-mer frequency is rotation-invariant).
- Ambiguous bases: k-mers containing non-ACGT are skipped.
- Cosine distance = 1 - cosine similarity.

Typical use
-----------
python addons/addon_03_reference_panel_projection.py ^
  --query_fasta data/combined_genomes.fasta ^
  --ref_fasta data/pmp_panel87.fasta ^
  --k 4

Safe to upload to GitHub (no hardcoded paths).
"""

import argparse
import csv
import gzip
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# -----------------------------
# Header richness heuristics
# -----------------------------
ACCESSION_RE = re.compile(r"^[A-Z]{1,4}\d{5,10}(\.\d+)?$")   # AJ534983.1, NC_001574.1
GENUS_SPECIES_RE = re.compile(r"\b[A-Z][a-z]{2,}\s+[a-z]{2,}\b")  # loose
KEYWORDS = ("country", "strain", "isolate", "host", "location", "collected", "isolation", "geo", "region")

# Conservative token extractors (keep identical to addon_02 behavior)
COUNTRY_PATTERNS = [
    r"\bGhana\b",
    r"\bTogo\b",
    r"\bNigeria\b",
    r"\bSri\s*Lanka\b",
    r"\bC[ôo]te\s*d['’]Ivoire\b",
    r"\bIvory\s*Coast\b",
    r"\bIvoire\b",
]
ISOLATE_PATTERNS = [
    r"\bisolate\s+([A-Za-z0-9_-]+)\b",
    r"\b(GH[A-Za-z0-9-]+)\b",
    r"\b(Gha[A-Za-z0-9-]+)\b",
    r"\b(CI[A-Za-z0-9-]+)\b",
    r"\b(NIG[A-Za-z0-9-]+)\b",
]


def opn(path: str):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.endswith(".gz") else \
        open(path, "rt", encoding="utf-8", errors="replace")


def fasta_iter(path: str) -> Iterable[Tuple[str, str, str]]:
    """
    Yield (seq_id, header_without_>, sequence_string) from FASTA.
    """
    sid = None
    header = None
    seq_chunks: List[str] = []
    with opn(path) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                # flush previous
                if sid is not None:
                    yield sid, header, "".join(seq_chunks)
                header = line[1:].strip()
                sid = header.split()[0] if header else ""
                seq_chunks = []
            else:
                if sid is None:
                    continue
                seq_chunks.append(line.strip())
        # flush last
        if sid is not None:
            yield sid, header, "".join(seq_chunks)


def header_features(full_header: str) -> Dict[str, str]:
    toks = full_header.split()
    sid = toks[0] if toks else ""
    rest = " ".join(toks[1:]) if len(toks) > 1 else ""
    low = full_header.lower()

    extra_tokens = max(0, len(toks) - 1)
    has_delims = any(c in full_header for c in ("|", ";", "/", "[", "]", "(", ")", "{", "}"))
    has_kw = any(k in low for k in KEYWORDS)
    has_species_like = bool(GENUS_SPECIES_RE.search(full_header))
    accession_like = bool(ACCESSION_RE.match(sid))

    score = extra_tokens
    score += 2 if has_delims else 0
    score += 2 if has_kw else 0
    score += 2 if has_species_like else 0
    score += 1 if accession_like else 0

    if score >= 6:
        level = "strong"
    elif score >= 2:
        level = "some"
    else:
        level = "id_only"

    return {
        "id": sid,
        "header": full_header,
        "extra_tokens": str(extra_tokens),
        "accession_like": str(int(accession_like)),
        "species_like": str(int(has_species_like)),
        "has_keywords": str(int(has_kw)),
        "has_delims": str(int(has_delims)),
        "score": str(score),
        "level": level,
        "provenance_text": rest,
    }


def write_header_table(fasta_path: str, out_tsv: str):
    rows = []
    ids = set()
    dup = 0
    for sid, hdr, _seq in fasta_iter(fasta_path):
        feat = header_features(hdr)
        rows.append(feat)
        if sid in ids:
            dup += 1
        ids.add(sid)

    if not rows:
        raise ValueError("No FASTA headers parsed; is this a valid FASTA?")

    with open(out_tsv, "wt", encoding="utf-8", newline="") as out:
        out.write("\t".join([
            "id", "level", "score", "extra_tokens", "accession_like", "species_like",
            "has_keywords", "has_delims", "provenance_text", "header"
        ]) + "\n")
        for r in rows:
            out.write("\t".join([
                r["id"], r["level"], r["score"], r["extra_tokens"], r["accession_like"], r["species_like"],
                r["has_keywords"], r["has_delims"],
                r["provenance_text"].replace("\t", " "),
                r["header"].replace("\t", " "),
            ]) + "\n")

    ctr = Counter(r["level"] for r in rows)
    total = len(rows)
    print(f"[OK] Header richness: {os.path.basename(fasta_path)}")
    print(f"     sequences: {total} (unique ids: {len(ids)}, dup ids: {dup})")
    for k in ("strong", "some", "id_only"):
        print(f"     {k:7s}: {ctr.get(k, 0)} ({ctr.get(k, 0)/total:.1%})")
    print(f"[OK] Wrote header table: {out_tsv}")


# -----------------------------
# k-mer cosine distance
# -----------------------------
_BASES = ("A", "C", "G", "T")
_BASE2BIT = {"A": 0, "C": 1, "G": 2, "T": 3}


def _kmer_indexer(k: int):
    """
    Return (mask, shift) for rolling 2-bit encoding.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    mask = (1 << (2 * k)) - 1
    shift = 2 * (k - 1)
    return mask, shift


def kmer_counts(seq: str, k: int) -> List[int]:
    """
    Count A/C/G/T k-mers using rolling 2-bit encoding.
    Skips any k-mer window containing non-ACGT.
    """
    seq = (seq or "").upper()
    n = len(seq)
    dim = 4 ** k
    out = [0] * dim
    if n < k:
        return out

    mask, _shift = _kmer_indexer(k)

    code = 0
    valid_run = 0

    for ch in seq:
        b = _BASE2BIT.get(ch, None)
        if b is None:
            code = 0
            valid_run = 0
            continue

        code = ((code << 2) | b) & mask
        valid_run += 1
        if valid_run >= k:
            out[code] += 1

    return out


def cosine_distance_from_counts(a: List[int], b: List[int]) -> float:
    """
    1 - (a·b)/(|a||b|). If either norm is 0, returns NaN.
    """
    dot = 0.0
    na = 0.0
    nb = 0.0
    # plain loop for speed; dim is 256 for k=4
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return 1.0 - (dot / (math.sqrt(na) * math.sqrt(nb)))


def extract_country(text: str) -> str:
    if not text:
        return ""
    hits = []
    for pat in COUNTRY_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            hits.append(m.group(0))

    norm = []
    for h in hits:
        h2 = re.sub(r"\s+", " ", h.strip())
        if re.search(r"ivoire|ivory", h2, flags=re.I):
            h2 = "Cote d'Ivoire"
        elif re.search(r"sri\s*lanka", h2, flags=re.I):
            h2 = "Sri Lanka"
        else:
            if h2.lower() in ("ghana", "togo", "nigeria"):
                h2 = h2.capitalize()
        norm.append(h2)

    seen = set()
    uniq = []
    for h in norm:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return ";".join(uniq)


def extract_isolate_tags(text: str) -> str:
    if not text:
        return ""
    tags = []
    for pat in ISOLATE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            if m.lastindex:
                tags.append(m.group(1))
            else:
                tags.append(m.group(0))

    tags = [re.sub(r"\s+", " ", t.strip()) for t in tags if t.strip()]
    seen = set()
    uniq = []
    for t in tags:
        t2 = t
        if t2.lower().startswith("isolate "):
            t2 = t2.split(None, 1)[1]
        if t2 not in seen:
            seen.add(t2)
            uniq.append(t2)
    return ";".join(uniq)


def main():
    ap = argparse.ArgumentParser(
        description="Project query genomes onto an external reference panel using k-mer cosine distance."
    )
    ap.add_argument("--query_fasta", required=True, help="Query genomes FASTA (e.g., core 48).")
    ap.add_argument("--ref_fasta", required=True, help="Reference panel FASTA (e.g., PMPP 86/87).")
    ap.add_argument("--k", type=int, default=4, help="k-mer size (default 4).")
    ap.add_argument("--out", default="projection_nearest_reference.tsv",
                    help="Output TSV mapping file.")
    ap.add_argument("--summary", default="projection_token_summary.tsv",
                    help="Token summary TSV file.")
    ap.add_argument("--ref_headers", default=None,
                    help="Optional: write reference header richness TSV to this path.")
    ap.add_argument("--query_headers", default=None,
                    help="Optional: write query header richness TSV to this path.")
    ap.add_argument("--keep_self", action="store_true",
                    help="If query and ref FASTA contain overlapping IDs, allow self-match. Default: exclude self if IDs equal.")
    ap.add_argument("--max_queries", type=int, default=0,
                    help="Debug: if >0, only process first N queries.")
    args = ap.parse_args()

    if args.k < 1 or args.k > 8:
        raise SystemExit("[ERR] --k must be between 1 and 8 (practical).")

    q_path = Path(args.query_fasta)
    r_path = Path(args.ref_fasta)
    if not q_path.exists():
        raise SystemExit(f"[ERR] missing --query_fasta: {q_path}")
    if not r_path.exists():
        raise SystemExit(f"[ERR] missing --ref_fasta: {r_path}")

    # Optional header richness tables
    if args.ref_headers:
        write_header_table(str(r_path), args.ref_headers)
    if args.query_headers:
        write_header_table(str(q_path), args.query_headers)

    # Load reference panel: headers + kmer counts
    print(f"[INFO] Loading reference panel: {r_path}")
    ref_ids: List[str] = []
    ref_headers: Dict[str, str] = {}
    ref_prov_text: Dict[str, str] = {}
    ref_level: Dict[str, str] = {}
    ref_score: Dict[str, str] = {}
    ref_counts: List[List[int]] = []

    seen_ref = set()
    for sid, hdr, seq in fasta_iter(str(r_path)):
        if not sid:
            continue
        if sid in seen_ref:
            # keep first; warn
            continue
        seen_ref.add(sid)

        feat = header_features(hdr)
        ref_ids.append(sid)
        ref_headers[sid] = hdr
        # provenance_text is "rest of header"; fall back to full header if blank
        prov = feat["provenance_text"] or feat["header"] or ""
        ref_prov_text[sid] = prov
        ref_level[sid] = feat["level"]
        ref_score[sid] = feat["score"]

        ref_counts.append(kmer_counts(seq, args.k))

    if not ref_ids:
        raise SystemExit("[ERR] reference FASTA had no usable records.")

    print(f"[OK] Reference loaded: {len(ref_ids)} sequences (k={args.k}, dim={4**args.k})")

    # Process queries and compute nearest ref
    print(f"[INFO] Projecting queries: {q_path}")
    out_fields = [
        "query_id",
        "nearest_ref_id",
        "distance",
        "nearest_ref_provenance_text",
        "nearest_ref_country_tag",
        "nearest_ref_isolate_tags",
        "nearest_ref_header_level",
        "nearest_ref_header_score",
    ]
    token_counter = Counter()

    n_q = 0
    written = 0
    with open(args.out, "wt", encoding="utf-8", newline="") as out_f:
        w = csv.DictWriter(out_f, delimiter="\t", fieldnames=out_fields)
        w.writeheader()

        for qid, qhdr, qseq in fasta_iter(str(q_path)):
            if not qid:
                continue
            n_q += 1
            if args.max_queries and n_q > args.max_queries:
                break

            q_counts = kmer_counts(qseq, args.k)

            best_id = ""
            best_d = float("inf")

            for rid, rc in zip(ref_ids, ref_counts):
                if (not args.keep_self) and (qid == rid):
                    continue
                d = cosine_distance_from_counts(q_counts, rc)
                if d != d:  # NaN
                    continue
                if d < best_d:
                    best_d = d
                    best_id = rid

            if not best_id or best_d == float("inf"):
                # still write row but empty
                w.writerow({
                    "query_id": qid,
                    "nearest_ref_id": "",
                    "distance": "",
                    "nearest_ref_provenance_text": "",
                    "nearest_ref_country_tag": "",
                    "nearest_ref_isolate_tags": "",
                    "nearest_ref_header_level": "",
                    "nearest_ref_header_score": "",
                })
                written += 1
                continue

            prov = ref_prov_text.get(best_id, "")
            country = extract_country(prov)
            isolates = extract_isolate_tags(prov)

            # token summary
            if country:
                for t in country.split(";"):
                    token_counter[t.lower()] += 1
            if isolates:
                for t in isolates.split(";"):
                    token_counter[t.lower()] += 1

            w.writerow({
                "query_id": qid,
                "nearest_ref_id": best_id,
                "distance": f"{best_d:.6g}",
                "nearest_ref_provenance_text": prov,
                "nearest_ref_country_tag": country,
                "nearest_ref_isolate_tags": isolates,
                "nearest_ref_header_level": ref_level.get(best_id, ""),
                "nearest_ref_header_score": ref_score.get(best_id, ""),
            })
            written += 1

    # Write token summary
    with open(args.summary, "wt", encoding="utf-8", newline="") as sum_f:
        w = csv.writer(sum_f, delimiter="\t")
        w.writerow(["token", "count"])
        for tok, cnt in token_counter.most_common():
            w.writerow([tok, cnt])

    print(f"[OK] Wrote: {args.out} (rows={written})")
    print(f"[OK] Wrote: {args.summary}")
    print("[DONE]")


if __name__ == "__main__":

    main()
