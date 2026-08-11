#!/usr/bin/env python3
"""Accession-level record, EVE-screening, and canonical ORF3 audit.

The screen is record-based. It does not experimentally establish episomal
status. Complete viral records of expected size and organization are retained;
host-genomic, partial endogenous, or host-flanking records are flagged.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO


def norm_acc(text: str) -> str:
    return str(text).strip().split()[0]


def load_records(folder: Path) -> Dict[str, object]:
    records = {}
    for p in sorted(folder.glob("*")):
        if not p.is_file() or p.suffix.lower() not in {".gb", ".gbk", ".genbank"}:
            continue
        if p.name.lower().startswith("combined_"):
            continue
        try:
            recs = list(SeqIO.parse(str(p), "genbank"))
        except Exception as exc:
            print(f"[WARN] Could not parse {p}: {exc}")
            continue
        for rec in recs:
            keys = {norm_acc(rec.id), norm_acc(rec.name), p.stem}
            for a in rec.annotations.get("accessions", []):
                keys.add(norm_acc(a))
            for key in keys:
                records.setdefault(key, rec)
                if "." in key:
                    records.setdefault(key.split(".")[0], rec)
    return records


def feature_text(feature) -> str:
    vals = []
    for key in ["gene", "product", "label", "standard_name", "note", "function"]:
        vals.extend(feature.qualifiers.get(key, []))
    return " | ".join(map(str, vals))


def all_qualifier_text(feature) -> str:
    if feature is None:
        return ""
    fields = []
    for key in sorted(feature.qualifiers):
        value = ";".join(map(str, feature.qualifiers.get(key, [])))
        fields.append(f"{key}={value}")
    return " | ".join(fields)


def feature_positions(feature, L: int) -> np.ndarray:
    mask = np.zeros(L, dtype=bool)
    parts = getattr(feature.location, "parts", [feature.location])
    for part in parts:
        start = int(part.start) % L
        end = int(part.end)
        if end <= L and start < end:
            mask[start:end] = True
        else:
            mask[start:L] = True
            mask[0:end % L] = True
    return mask


def predicted_positions(row: pd.Series, L: int) -> np.ndarray:
    mask = np.zeros(L, dtype=bool)
    start = int(row["start0"]) % L
    nt_len = int(row["nt_len"])
    idx = (start + np.arange(nt_len)) % L
    mask[idx] = True
    return mask


def select_orf3_feature(record):
    candidates = []
    for feature in record.features:
        if str(feature.type).upper() != "CDS":
            continue
        text = feature_text(feature)
        low = text.lower()
        length = int(len(feature.location))
        score = length / 10000.0
        if re.search(r"\borf\s*3\b|\borf3\b", low):
            score += 100
        if "polyprotein" in low:
            score += 50
        for token in ["reverse transcriptase", "rnase h", "movement protein", "capsid", "coat protein", "aspartic protease"]:
            if token in low:
                score += 5
        candidates.append((score, length, feature, text))
    if not candidates:
        return None, "", "no CDS feature"
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    score, _, feature, text = candidates[0]
    evidence = "explicit ORF3/polyprotein annotation" if score >= 50 else "longest/most informative CDS fallback"
    return feature, text, evidence




def source_feature_for_record(record):
    return next((f for f in record.features if str(f.type).lower() == "source"), None)


def record_organism(record) -> str:
    """Return the organism from annotations, falling back to /source qualifiers."""
    value = str(record.annotations.get("organism", "") or "").strip()
    if value:
        return value
    source_feature = source_feature_for_record(record)
    if source_feature is not None:
        vals = source_feature.qualifiers.get("organism", [])
        if vals:
            return str(vals[0]).strip()
    return ""


def record_text(record) -> str:
    vals = [record.description, record.annotations.get("organism", ""), record.annotations.get("source", "")]
    vals.extend(record.annotations.get("keywords", []) or [])
    vals.append(record.annotations.get("comment", ""))
    for feature in record.features:
        if feature.type == "source":
            vals.append(feature_text(feature))
            for v in feature.qualifiers.values():
                vals.extend(map(str, v))
    return " | ".join(map(str, vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genbank_dir", required=True)
    ap.add_argument("--predicted_orfs", required=True)
    ap.add_argument("--accessions", required=True)
    ap.add_argument("--species_table", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_expected_length", type=int, default=6500)
    ap.add_argument("--max_expected_length", type=int, default=8000)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    accessions = [x.strip() for x in Path(args.accessions).read_text().splitlines() if x.strip() and not x.startswith("#")]
    records = load_records(Path(args.genbank_dir).resolve())
    pred = pd.read_csv(args.predicted_orfs)
    p_name = "name" if "name" in pred.columns else "genome"
    pred[p_name] = pred[p_name].astype(str)
    pred_long = pred.sort_values("aa_len", ascending=False).drop_duplicates(p_name).set_index(p_name)
    species = None
    if args.species_table:
        species = pd.read_csv(args.species_table, sep="\t", dtype=str).drop_duplicates("accession").set_index("accession")

    record_rows = []
    orf_rows = []
    merged_rows = []
    eve_terms = re.compile(r"\bendogenous\b|\bintegrated\b|integration|endogenous viral element|\beve\b", re.I)
    flank_terms = re.compile(r"host[- ]flank|flanking host|theobroma cacao chromosome|host genomic|genomic scaffold", re.I)
    assembly_terms = re.compile(r"chromosome|scaffold|whole genome shotgun|genomic contig", re.I)

    for accession in accessions:
        rec = records.get(accession) or records.get(accession.split(".")[0])
        species_row = species.loc[accession].to_dict() if species is not None and accession in species.index else {}
        if rec is None:
            record_rows.append({"accession": accession, "record_available": False, "inclusion_status": "annotation record missing"})
            orf_rows.append({"accession": accession, "record_available": False, "orf3_audit_status": "annotation record missing"})
            continue
        L = len(rec.seq)
        text = record_text(rec)
        organism = record_organism(rec)
        topology = str(rec.annotations.get("topology", ""))
        complete = "complete genome" in str(rec.description).lower()
        length_ok = args.min_expected_length <= L <= args.max_expected_length
        eve_flag = bool(eve_terms.search(text))
        flank_flag = bool(flank_terms.search(text))
        assembly_flag = bool(assembly_terms.search(text)) and "virus" not in organism.lower()
        viral_record = "virus" in organism.lower() or "badnavirus" in organism.lower()
        source_feature = source_feature_for_record(rec)
        source_qual = all_qualifier_text(source_feature)
        inclusion = "included complete viral record"
        if eve_flag or flank_flag or assembly_flag or not length_ok or not complete or not viral_record:
            inclusion = "manual review required"
        record_row = {
            "accession": accession,
            **species_row,
            "record_available": True,
            "record_description": rec.description,
            "record_organism": organism,
            "record_length_bp": L,
            "record_topology": topology,
            "complete_genome_description": complete,
            "expected_badnavirus_length": length_ok,
            "viral_record_organism": viral_record,
            "eve_keyword_flag": eve_flag,
            "host_flanking_keyword_flag": flank_flag,
            "host_assembly_keyword_flag": assembly_flag,
            "source_qualifiers": source_qual,
            "inclusion_status": inclusion,
            "episomal_status_statement": "not experimentally established by record-based screening",
        }
        record_rows.append(record_row)

        feature, ann_text, evidence = select_orf3_feature(rec)
        prow = pred_long.loc[accession] if accession in pred_long.index else None
        orf_row = {"accession": accession, "record_available": True, "annotated_orf3_text": ann_text, "annotated_orf3_selection_basis": evidence}
        if feature is None or prow is None:
            orf_row.update({
                "predicted_longest_orf_available": prow is not None,
                "annotated_orf3_available": feature is not None,
                "orf3_audit_status": "manual review required",
            })
        else:
            ann_mask = feature_positions(feature, L)
            pred_mask = predicted_positions(prow, L)
            overlap = int(np.sum(ann_mask & pred_mask))
            ann_len = int(np.sum(ann_mask)); pred_len = int(np.sum(pred_mask)); union = int(np.sum(ann_mask | pred_mask))
            ann_aa = None
            translations = feature.qualifiers.get("translation", [])
            if translations:
                ann_aa = len(str(translations[0]).replace("*", ""))
            else:
                ann_aa = ann_len // 3
            same_strand = int(getattr(feature.location, "strand", 0) or 0) == (1 if str(prow["strand"]) == "+" else -1)
            frac_pred = overlap / pred_len if pred_len else np.nan
            frac_ann = overlap / ann_len if ann_len else np.nan
            consistent = bool(same_strand and frac_pred >= 0.80 and frac_ann >= 0.80)
            orf_row.update({
                "predicted_longest_orf_available": True,
                "annotated_orf3_available": True,
                "predicted_start0": int(prow["start0"]),
                "predicted_end0": int(prow["end0"]),
                "predicted_wrap": bool(prow["wrap"]),
                "predicted_nt_len": int(prow["nt_len"]),
                "predicted_aa_len": int(prow["aa_len"]),
                "annotated_start0": int(feature.location.start),
                "annotated_end0": int(feature.location.end),
                "annotated_strand": int(getattr(feature.location, "strand", 0) or 0),
                "annotated_nt_len": ann_len,
                "annotated_aa_len": ann_aa,
                "same_strand": same_strand,
                "overlap_nt": overlap,
                "overlap_fraction_of_predicted": frac_pred,
                "overlap_fraction_of_annotated": frac_ann,
                "coordinate_jaccard": overlap / union if union else np.nan,
                "consistent_with_annotated_orf3": consistent,
                "orf3_audit_status": "consistent" if consistent else "exception/manual review",
            })
        orf_rows.append(orf_row)

    record_df = pd.DataFrame(record_rows)
    orf_df = pd.DataFrame(orf_rows)
    record_df.to_csv(out / "accession_record_and_eve_screen.csv", index=False)
    orf_df.to_csv(out / "orf3_annotation_audit.csv", index=False)
    inclusion = record_df.merge(orf_df, on=["accession", "record_available"], how="outer", suffixes=("_record", "_orf3"))
    inclusion.to_csv(out / "accession_inclusion_and_orf3_audit.csv", index=False)
    pd.DataFrame([{
        "n_accessions_expected": len(accessions),
        "n_genbank_records_available": int(record_df.get("record_available", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_records_flagged_for_manual_review": int((record_df.get("inclusion_status", pd.Series(dtype=str)) == "manual review required").sum()),
        "n_predicted_longest_orfs_consistent_with_annotated_orf3": int(orf_df.get("consistent_with_annotated_orf3", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_orf3_exceptions_or_manual_review": int((orf_df.get("orf3_audit_status", pd.Series(dtype=str)) == "exception/manual review").sum()),
        "screen_scope": "record-based screening; not experimental episomal-status determination",
    }]).to_csv(out / "accession_orf3_eve_audit_summary.csv", index=False)
    print(f"[OK] Accession/ORF3/EVE audit outputs: {out}")


if __name__ == "__main__":
    main()
