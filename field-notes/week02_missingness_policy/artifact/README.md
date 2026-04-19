# Week 2 Artifact — Missingness Policy and Impact

This artifact builds a deterministic demo dataset with structured missingness and exports a policy-aware audit trail.

## Run

From the repository root:

```bash
python field-notes/week02_missingness_policy/artifact/scripts/build_week02_missingness_artifacts.py
```

## Outputs

Stored in `field-notes/week02_missingness_policy/artifact/outputs/week02_missingness_audit/`:

- `qa/missingness_report.json`
- `qa/co_missingness_report.json`
- `qa/missingness_tracking.json`
- `qa/cleaning_log.json`
- `eda/missingness_heatmap.png`
- `manifest.json`

## Notes

- `age`, `stage`, and `biomarker_x` are the tracked missingness fields.
- `silent_row_drop` is explicitly set to `false`.
- The goal is visibility, not premature deletion or imputation.
