# RATEM

**Regime-Aware Time-to-Event Modeling Pipeline**

RATEM is a statistics-first build-in-public project for **reproducible time-to-event analysis**.
The current repository covers the first four field-note modules:

1. **Week 1 — Validation / Dataset as a Contract**
2. **Week 2 — Missingness Policy and Impact**
3. **Week 3 — Kaplan–Meier + At-Risk Table**
4. **Week 4 — Stratified Kaplan–Meier**

The governing principle is:

**Assumptions → Evidence → Artifacts**

---

## What this repository is for

RATEM is designed as a methodological framework for trustworthy survival analysis.
It prioritizes:

- explicit assumptions
- auditable preprocessing
- policy-aware missingness handling
- correct treatment of censoring
- reproducible survival artifacts
- descriptive subgroup comparison before regression

This is a **method demonstration repository**. It does **not** provide medical or financial advice.

---

## Current implemented scope (Weeks 1–4)

### Week 1 — Validation
- builds an intentionally flawed demo dataset from `data/raw/VA.csv`
- runs contract-style validation checks
- exports:
  - `schema_report.json`
  - `missingness_report.json`
  - `cleaning_log.json`
  - `manifest.json`

### Week 2 — Missingness
- creates a deterministic demo dataset with structured missingness
- exports a policy-aware missingness audit
- exports:
  - variable-wise missingness report
  - co-missingness report
  - tracking by covariate / outcome / follow-up band
  - missingness heatmap
  - cleaning log
  - manifest

### Week 3 — Kaplan–Meier baseline
- reuses the Week 2 analysis-ready demo dataset
- exports an overall Kaplan–Meier curve with censor ticks and an at-risk table
- exports:
  - `cleaned.csv`
  - `km_overall.png`
  - `km_event_table.csv`
  - `km_overall_summary.json`
  - `cleaning_log.json`
  - `manifest.json`

### Week 4 — Stratified Kaplan–Meier
- reuses the Week 3 analysis dataset
- exports subgroup survival curves by:
  - `treatment_arm`
  - `stage`
- exports:
  - `km_by_group_treatment_arm.png`
  - `km_by_group_stage.png`
  - group summaries
  - stratification log
  - manifest

---

## Repository structure

```text
RATEM/
├── data/
│   ├── raw/
│   └── demo/
├── docs/
├── field-notes/
│   ├── week01_dataset_contract/
│   ├── week02_missingness_policy/
│   ├── week03_kaplan_meier_baseline/
│   └── week04_stratified_km/
└── scripts/
```

---

## Quick start

Create the environment:

```bash
conda env create -f environment.yml
conda activate ratem
```

Run the implemented artifact scripts from the repository root:

```bash
python field-notes/week01_dataset_contract/artifact/scripts/prepare_week01_data.py
python field-notes/week01_dataset_contract/artifact/scripts/run_validation.py
python field-notes/week02_missingness_policy/artifact/scripts/build_week02_missingness_artifacts.py
python field-notes/week03_kaplan_meier_baseline/artifact/scripts/build_week03_km_artifacts.py
python field-notes/week04_stratified_km/artifact/scripts/build_week04_stratified_km_artifacts.py
```

---

## Data note

The raw event-time backbone comes from the Veterans' Administration lung cancer trial dataset in `data/raw/VA.csv`.
For public method demonstrations, the repo also constructs synthetic fields such as `stage` and `biomarker_x` so validation, missingness, and subgroup examples can be shown consistently.

---

## Why the first four weeks matter

The first four field notes establish the baseline survival-analysis layer:

- validate the dataset before modeling
- audit missingness before deletion or imputation
- use Kaplan–Meier for censored descriptive survival
- separate subgroup description from adjusted inference

That is the minimal foundation before Cox modeling, PH diagnostics, and regime-aware extensions.

---

## Status

Implemented now:
- validation
- missingness audit
- Kaplan–Meier baseline
- stratified Kaplan–Meier

Planned next:
- Cox summary export
- proportional hazards diagnostics
- regime-aware modeling
- evaluation and calibration
- uncertainty quantification

