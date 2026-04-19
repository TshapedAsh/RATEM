from __future__ import annotations

from pathlib import Path
import sys
import json

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ratem_demo_utils import (  # noqa: E402
    build_base_demo_dataset,
    inject_policy_aware_missingness,
    manifest_payload,
    save_dataframe,
    write_json,
)
from scripts.ratem_survival_utils import plot_km_with_at_risk  # noqa: E402


RUN_ID = "week04_stratified_km"
RAW_DATA = REPO_ROOT / "data" / "raw" / "VA.csv"
ANALYSIS_DATA = REPO_ROOT / "data" / "demo" / "ratem_demo_analysis.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / RUN_ID
DATA_DIR = OUTPUT_DIR / "data"
QA_DIR = OUTPUT_DIR / "qa"
SURVIVAL_DIR = OUTPUT_DIR / "survival"


def _analysis_input() -> pd.DataFrame:
    if ANALYSIS_DATA.exists():
        return pd.read_csv(ANALYSIS_DATA, dtype={"id": str})

    base_df = build_base_demo_dataset(RAW_DATA)
    missing_df = inject_policy_aware_missingness(base_df)
    save_dataframe(missing_df, ANALYSIS_DATA)
    return missing_df


def _summary_with_exclusions(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
    before_rows = len(df)
    subset = df.dropna(subset=[column]).copy()
    payload = {
        "stratification_variable": column,
        "rows_before_dropna": int(before_rows),
        "rows_used": int(len(subset)),
        "rows_excluded_for_missing_group": int(before_rows - len(subset)),
    }
    return subset, payload


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    SURVIVAL_DIR.mkdir(parents=True, exist_ok=True)

    analysis_df = _analysis_input()
    save_dataframe(analysis_df, DATA_DIR / "cleaned.csv")

    treatment_df, treatment_log = _summary_with_exclusions(analysis_df, "treatment_arm")
    stage_df, stage_log = _summary_with_exclusions(analysis_df, "stage")

    treatment_summaries = plot_km_with_at_risk(
        treatment_df,
        SURVIVAL_DIR / "km_by_group_treatment_arm.png",
        title="Stratified Kaplan–Meier by treatment arm",
        time_unit="days",
        group_col="treatment_arm",
        group_order=["A", "B"],
    )
    stage_summaries = plot_km_with_at_risk(
        stage_df,
        SURVIVAL_DIR / "km_by_group_stage.png",
        title="Stratified Kaplan–Meier by stage",
        time_unit="days",
        group_col="stage",
        group_order=["I", "II", "III", "IV"],
    )

    write_json(QA_DIR / "stratification_log.json", [treatment_log, stage_log])
    write_json(
        SURVIVAL_DIR / "km_group_summaries_treatment_arm.json",
        treatment_summaries,
    )
    write_json(
        SURVIVAL_DIR / "km_group_summaries_stage.json",
        stage_summaries,
    )

    output_files = [
        DATA_DIR / "cleaned.csv",
        QA_DIR / "stratification_log.json",
        SURVIVAL_DIR / "km_by_group_treatment_arm.png",
        SURVIVAL_DIR / "km_by_group_stage.png",
        SURVIVAL_DIR / "km_group_summaries_treatment_arm.json",
        SURVIVAL_DIR / "km_group_summaries_stage.json",
    ]
    manifest = manifest_payload(
        run_id=RUN_ID,
        input_files={
            "raw_va": RAW_DATA,
            "analysis_dataset": ANALYSIS_DATA,
        },
        output_files=output_files,
        notes={
            "artifact": "Stratified Kaplan–Meier subgroup comparisons",
            "descriptive_only": True,
            "warning": "Subgroup curves are not adjusted effect estimates.",
        },
    )
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    preview = {
        "treatment_groups": {
            key: value for key, value in treatment_summaries.items() if key != "time_grid"
        },
        "stage_groups": {
            key: value for key, value in stage_summaries.items() if key != "time_grid"
        },
    }
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
