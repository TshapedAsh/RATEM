# Week 4 Artifact — Stratified Kaplan–Meier

This artifact exports subgroup-specific Kaplan–Meier curves.

## Run

From the repository root:

```bash
python field-notes/week04_stratified_km/artifact/scripts/build_week04_stratified_km_artifacts.py
```

## Outputs

Stored in `field-notes/week04_stratified_km/artifact/outputs/week04_stratified_km/`:

- `data/cleaned.csv`
- `qa/stratification_log.json`
- `survival/km_by_group_treatment_arm.png`
- `survival/km_by_group_stage.png`
- `survival/km_group_summaries_treatment_arm.json`
- `survival/km_group_summaries_stage.json`
- `manifest.json`

## Interpretation reminder

- subgroup curves improve descriptive structure
- subgroup curves are not adjusted effect estimates
- curve crossing is information, not nuisance
- group-wise at-risk counts matter for tail interpretation
