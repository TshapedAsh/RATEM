# Week 3 Artifact — Kaplan–Meier + At-Risk Table

This artifact exports the baseline descriptive survival view used after censoring semantics are mapped correctly.

## Run

From the repository root:

```bash
python field-notes/week03_kaplan_meier_baseline/artifact/scripts/build_week03_km_artifacts.py
```

## Outputs

Stored in `field-notes/week03_kaplan_meier_baseline/artifact/outputs/week03_km_baseline/`:

- `data/cleaned.csv`
- `qa/cleaning_log.json`
- `survival/km_overall.png`
- `survival/km_event_table.csv`
- `survival/km_overall_summary.json`
- `manifest.json`

## Interpretation reminder

- steps correspond to observed events
- censor ticks are not events
- the at-risk table keeps the denominator visible
- this is descriptive survival analysis, not causal inference
