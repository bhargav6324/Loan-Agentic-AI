# Reproducibility Manifest

*Aligned with selected good-practice principles from the NIST AI RMF, ISO/IEC 5259, and Datasheets for Datasets.*

## Environment and Provenance
- **Dataset Checksum (SHA-256):** `669ff7aeda69a60200732a2b22b7b9cf46bb38681c10892bfb68cc2c38f75f1a`
- **Generator Checksum (SHA-256):** `bdcb361730e59d193817b0d2f4eef7f7db2e28f998e02de00364f289490e7892` (`research/data/generate_dataset.py`)
- **Python Version:** 3.12.3
- **Generation Seed:** 42 (Hardcoded in generator)
- **Dataset Size:** 5,000 records
- **Target Prevalence:** 11.34% defaults
- **Source-Code Commit Identifier:** unavailable in the uploaded archive (the archive contains no `.git` metadata)

## Validation Signature
- **Validation Script:** `dataset_validation.py`
- **Status:** PASSED (Zero critical schema or leakage constraints violated)
- **Output:** Checked and compiled into `validation_results.json` and `DATA_QUALITY_REPORT.md`.
