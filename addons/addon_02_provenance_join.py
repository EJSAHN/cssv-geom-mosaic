#!/usr/bin/env python
# addon_02_provenance_join.py
"""
Compute nearest-neighbor (within-dataset) from a distance matrix and join
provenance information parsed from FASTA headers.

Inputs:
  --dist     Distance matrix CSV (square; first column may be row names)
  --headers  TSV produced by addon_01_reference_projection.py (id + provenance_text)
Outputs:
  --out      TSV table: query -> nearest neighbor + distance + provenance fields
  --summary  TSV token frequency summary (optional)

Works on Windows/macOS/Linux. No hardcoded directories.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# --- simple token extractors (conservative; avoids over-claiming) ---
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
    r"\bisolate\s+([A-Za-z0-9_-]+)\b",      # "isolate X"
    r"\b(GH[A-Za-z0-9-]+)\b",               # GH67, GH64 ...
    r"\b(Gha[A-Za-z0-9-]+)\b",              # Gha37-15 ...
    r"\b(CI[A-Za-z0-9-]+)\b",               # CI152-09 ...
    r"\b(NIG[A-Za-z0-9-]+)\b",              # NIG5 ...
]


def read_headers_tsv(path: Path) -> Dict[str, Dict[str, str]]:
    """Return dict: id -> row dict (must include provenance_text if present)."""
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"[headers] missing required columns: {sorted(missing)}")

        for row in reader:
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            out[rid] = {k: (v.strip() if isinstance(v, str) else ("" if v is None else str(v)))
                        for k, v in row.items()}
    return out


def normalize_id(x: str) -> str:
    return (x or "").strip()


def parse_distance_matrix(path: Path) -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Read a square distance matrix CSV.

    Supports:
      - header row = column ids
      - first column may be row ids (often named 'Unnamed: 0')

    Returns:
      col_ids, row_ids, matrix (row-major floats)
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows) < 2:
        raise ValueError("[dist] file looks empty or too small")

    header = rows[0]
    data_rows = rows[1:]

    # Determine if first cell is blank / label; and whether first column is row names
    # Common pattern: header[0] == '' or 'Unnamed: 0'
    col_ids = [normalize_id(h) for h in header]

    # If first header cell is blank-ish, treat remaining header cells as column ids
    if col_ids[0] in ("", "unnamed: 0", "unnamed:0", "row", "id"):
        col_ids = col_ids[1:]
        has_row_names = True
    else:
        # might still have row names if first column values are non-numeric
        has_row_names = True  # assume yes; verify below

    row_ids: List[str] = []
    matrix: List[List[float]] = []

    for r in data_rows:
        if not r:
            continue

        if has_row_names:
            rid = normalize_id(r[0])
            vals = r[1:]
        else:
            rid = ""
            vals = r

        # If we assumed row names but rid is numeric and vals length mismatch, fallback
        if has_row_names and rid and _looks_numeric(rid) and len(vals) == len(col_ids) - 1:
            # actually no row names; rebuild
            has_row_names = False
            rid = ""
            vals = r
            row_ids.clear()
            matrix.clear()
            # restart parsing in a simple way
            return parse_distance_matrix_no_row_names(path)

        row_ids.append(rid if rid else f"row{len(row_ids)+1}")

        row_floats: List[float] = []
        for v in vals:
            v = (v or "").strip()
            if v == "":
                row_floats.append(float("nan"))
            else:
                try:
                    row_floats.append(float(v))
                except ValueError:
                    row_floats.append(float("nan"))
        matrix.append(row_floats)

    # sanity
    if len(matrix) != len(row_ids):
        raise ValueError("[dist] internal parse error (row count mismatch)")

    # If has_row_names pattern: matrix should be len(row_ids) x len(col_ids)
    # If header had blank cell and we removed it, this should align.
    if matrix and col_ids and len(matrix[0]) != len(col_ids):
        raise ValueError(
            f"[dist] shape mismatch: got {len(matrix)}x{len(matrix[0])}, "
            f"but header has {len(col_ids)} columns. "
            f"Check if this is a square matrix with row/col ids."
        )

    return col_ids, row_ids, matrix


def parse_distance_matrix_no_row_names(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = [normalize_id(h) for h in rows[0]]
    col_ids = header
    row_ids = [f"row{i+1}" for i in range(len(rows)-1)]
    matrix = []
    for r in rows[1:]:
        row_floats = []
        for v in r:
            v = (v or "").strip()
            row_floats.append(float(v) if v else float("nan"))
        matrix.append(row_floats)
    return col_ids, row_ids, matrix


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def nearest_neighbor(row_ids: List[str], col_ids: List[str], mat: List[List[float]]) -> List[Tuple[str, str, float]]:
    """
    For each row, find nearest neighbor column (minimum distance), excluding self match if ids equal.
    Returns list of (query_id, nn_id, nn_dist).
    """
    out = []
    col_index = {cid: j for j, cid in enumerate(col_ids)}

    for i, q in enumerate(row_ids):
        best_id = ""
        best_val = float("inf")

        for j, cid in enumerate(col_ids):
            v = mat[i][j]
            if v != v:  # NaN
                continue
            if q == cid:
                continue
            if v < best_val:
                best_val = v
                best_id = cid

        if best_id == "" or best_val == float("inf"):
            out.append((q, "", float("nan")))
        else:
            out.append((q, best_id, best_val))
    return out


def extract_country(text: str) -> str:
    if not text:
        return ""
    hits = []
    for pat in COUNTRY_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            hits.append(m.group(0))
    # normalize common variants
    norm = []
    for h in hits:
        h2 = re.sub(r"\s+", " ", h.strip())
        # unify "Ivoire"/"Cote d'Ivoire"/"Ivory Coast" loosely
        if re.search(r"ivoire|ivory", h2, flags=re.I):
            h2 = "Cote d'Ivoire"
        elif re.search(r"sri\s*lanka", h2, flags=re.I):
            h2 = "Sri Lanka"
        else:
            # Title-case Ghana/Togo/Nigeria
            if h2.lower() in ("ghana", "togo", "nigeria"):
                h2 = h2.capitalize()
        norm.append(h2)
    # unique, stable order
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
            if m.lastindex:  # captured group
                tags.append(m.group(1))
            else:
                tags.append(m.group(0))
    # normalize spacing
    tags = [re.sub(r"\s+", " ", t.strip()) for t in tags if t.strip()]
    # uniq preserve order
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
        description="Join nearest-neighbor IDs from a distance matrix with provenance text from headers table."
    )
    ap.add_argument("--dist", required=True, help="Distance matrix CSV (square).")
    ap.add_argument("--headers", required=True, help="Headers richness TSV from addon_01_reference_projection.py")
    ap.add_argument("--out", default="nearest_neighbor_provenance.tsv", help="Output TSV filename.")
    ap.add_argument("--summary", default="nearest_neighbor_token_summary.tsv", help="Token summary TSV filename.")
    args = ap.parse_args()

    dist_path = Path(args.dist)
    hdr_path = Path(args.headers)
    out_path = Path(args.out)
    sum_path = Path(args.summary)

    if not dist_path.exists():
        raise SystemExit(f"[ERR] missing --dist file: {dist_path}")
    if not hdr_path.exists():
        raise SystemExit(f"[ERR] missing --headers file: {hdr_path}")

    headers = read_headers_tsv(hdr_path)
    col_ids, row_ids, mat = parse_distance_matrix(dist_path)

    nn = nearest_neighbor(row_ids, col_ids, mat)

    # write main table
    fieldnames = [
        "query_id",
        "nearest_id",
        "distance",
        "nearest_provenance_text",
        "nearest_country_tag",
        "nearest_isolate_tags",
        "nearest_header_level",
        "nearest_header_score",
    ]

    token_counter = Counter()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()

        for q, ref, d in nn:
            ref_row = headers.get(ref, {})
            prov = ref_row.get("provenance_text", "") or ref_row.get("header", "") or ""
            country = extract_country(prov)
            isolates = extract_isolate_tags(prov)

            # token summary (very conservative)
            if country:
                for t in country.split(";"):
                    token_counter[t.lower()] += 1
            if isolates:
                for t in isolates.split(";"):
                    token_counter[t.lower()] += 1

            w.writerow({
                "query_id": q,
                "nearest_id": ref,
                "distance": f"{d:.6g}" if d == d else "",
                "nearest_provenance_text": prov,
                "nearest_country_tag": country,
                "nearest_isolate_tags": isolates,
                "nearest_header_level": ref_row.get("level", ""),
                "nearest_header_score": ref_row.get("score", ""),
            })

    # write token summary
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["token", "count"])
        for tok, cnt in token_counter.most_common():
            w.writerow([tok, cnt])

    print("[OK] Wrote:", out_path)
    print("[OK] Wrote:", sum_path)
    print("[DONE]")


if __name__ == "__main__":
    main()