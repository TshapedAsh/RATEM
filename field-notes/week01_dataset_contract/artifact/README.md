# Week 1 Artifact — Validation Run

This artifact creates an intentionally flawed demo dataset and runs contract-style validation checks.

## Run

From the repository root:

```bash
python field-notes/week01_dataset_contract/artifact/scripts/prepare_week01_data.py
python field-notes/week01_dataset_contract/artifact/scripts/run_validation.py
```

## Outputs

Stored in `field-notes/week01_dataset_contract/artifact/outputs/`:

- `schema_report.json`
- `missingness_report.json`
- `cleaning_log.json`
- `manifest.json`

## Purpose

The dataset is supposed to fail.
The goal is to make silent failure visible before any modeling begins.
