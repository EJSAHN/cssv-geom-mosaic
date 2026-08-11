# Input-preparation tools

## `cssv_prepare_core_inputs.py`

Stages the exact accessions listed in `ACCESSIONS.txt` as one FASTA and one GenBank record per file.

Search order:
1. accession-specific files in one or more local source folders;
2. an optional multi-FASTA reference panel;
3. NCBI Entrez for records still missing when `--download_missing` is specified.

```bash
python tools/cssv_prepare_core_inputs.py \
  --accessions ACCESSIONS.txt \
  --source_dir "path/to/local/sequences" \
  --panel_fasta "path/to/reference_panel.fasta" \
  --out_fasta_dir "data/core48_fasta" \
  --out_genbank_dir "data/core48_genbank" \
  --email "your.email@example.org" \
  --download_missing
```

Outputs:
- accession-specific FASTA files in `--out_fasta_dir`;
- accession-specific GenBank files in `--out_genbank_dir`;
- `combined_core48.fasta` and `combined_core48.gb` in the parent input folder;
- `input_manifest.tsv` documenting accession, source, path, and sequence length.

Combined files are deliberately written outside the accession-specific input directories so the core pipeline cannot load them as duplicate records.
