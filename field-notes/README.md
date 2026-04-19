# RATEM Field Notes

This directory contains the public-facing **RATEM Field Notes** build-in-public series.

Each implemented week has two layers:

- a **concept** folder for the statistical lesson
- an **artifact** folder for the reproducible code and exported outputs

## Implemented weeks

```text
field-notes/
├── week01_dataset_contract/
├── week02_missingness_policy/
├── week03_kaplan_meier_baseline/
└── week04_stratified_km/
```

## Current logic arc

1. **Validation** — the dataset must satisfy an explicit contract.
2. **Missingness** — blanks are part of the data-generating process, not just noise.
3. **Kaplan–Meier baseline** — censored time-to-event data needs survival-aware descriptive plotting.
4. **Stratified Kaplan–Meier** — subgroup curves improve description but do not provide adjusted causal claims.

## Build rule

No field note should exist without code or exported artifacts to back it.
