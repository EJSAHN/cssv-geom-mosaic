# Changelog

## 1.0.0

- Implemented full circular extraction for origin-spanning barcode windows.
- Included the last-to-first barcode boundary in circular switch counts and switch rates.
- Normalized barcode entropy by `log2(K)`, where `K` is the total number of barcode clusters.
- Defined the public ranking as a three-component `mosaic_complexity_score`: low dominant-label purity, normalized barcode entropy, and switch rate.
- Retained ORF–mosaic agreement and mismatch as separate diagnostic outputs rather than score components.
- Added published-species validation, MDS diagnostics, barcode K/stability diagnostics, ORF3 neighbor-joining bootstrap support, ORF-boundary threshold sensitivity, and accession-level ORF3/EVE record audits.
- Added accession-exact input preparation and a full analysis/validation driver.
- Improved MAFFT handling across native and Windows Subsystem for Linux environments.
- Ensured CSV-only analysis checks create their output directories without requiring plotting.
