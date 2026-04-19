# RATEM GitHub Sync — Weeks 1 to 4

This document maps the first four LinkedIn field notes to concrete repository artifacts.

## Week 1 — Dataset as a Contract

**Concept**
- idea: validation before optimism
- repo role: explicit validation demo

**Artifact**
- script: `field-notes/week01_dataset_contract/artifact/scripts/`
- outputs: `schema_report.json`, `missingness_report.json`, `cleaning_log.json`, `manifest.json`

## Week 2 — Missingness Policy and Impact

**Concept**
- idea: missingness is a mechanism, not noise
- repo role: policy-aware audit layer

**Artifact**
- script: `field-notes/week02_missingness_policy/artifact/scripts/build_week02_missingness_artifacts.py`
- outputs:
  - `qa/missingness_report.json`
  - `qa/co_missingness_report.json`
  - `qa/missingness_tracking.json`
  - `qa/cleaning_log.json`
  - `eda/missingness_heatmap.png`
  - `manifest.json`

## Week 3 — Kaplan–Meier + At-Risk Table

**Concept**
- idea: censoring changes the estimand
- repo role: establish the correct descriptive survival baseline

**Artifact**
- script: `field-notes/week03_kaplan_meier_baseline/artifact/scripts/build_week03_km_artifacts.py`
- outputs:
  - `data/cleaned.csv`
  - `survival/km_overall.png`
  - `survival/km_event_table.csv`
  - `survival/km_overall_summary.json`
  - `qa/cleaning_log.json`
  - `manifest.json`

## Week 4 — Stratified Kaplan–Meier

**Concept**
- idea: hazard, survival, and event probability are not interchangeable
- repo role: keep estimands distinct before regression language appears

**Artifact**
- script: `field-notes/week04_stratified_km/artifact/scripts/build_week04_stratified_km_artifacts.py`
- outputs:
  - `survival/km_by_group_treatment_arm.png`
  - `survival/km_by_group_stage.png`
  - `survival/km_group_summaries_treatment_arm.json`
  - `survival/km_group_summaries_stage.json`
  - `qa/stratification_log.json`
  - `manifest.json`

## Why this sync matters

The public content plan requires each concept post to be paired with an implemented artifact in the same week.
This repo state is the minimum GitHub baseline needed to make Weeks 1–4 visibly real.
