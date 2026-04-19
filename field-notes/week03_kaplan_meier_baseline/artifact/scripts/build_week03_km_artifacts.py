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
from scripts.ratem_survival_utils import km_event_table, plot_km_with_at_risk  # noqa: E402


RUN_ID = "week03_km_baseline"
RAW_DATA = REPO_ROOT / "data" / "raw" / "VA.csv"
INPUT_DATA = REPO_ROOT / "data" / "demo" / "ratem_demo_missingness.csv"
ANALYSIS_DATA = REPO_ROOT / "data" / "demo" / "ratem_demo_analysis.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / RUN_ID
DATA_DIR = OUTPUT_DIR / "data"
QA_DIR = OUTPUT_DIR / "qa"
SURVIVAL_DIR = OUTPUT_DIR / "survival"


def _build_input_if_missing() -> pd.DataFrame:
    if INPUT_DATA.exists():
        return pd.read_csv(INPUT_DATA, dtype={"id": str})

    base_df = build_base_demo_dataset(RAW_DATA)
    missing_df = inject_policy_aware_missingness(base_df)
    save_dataframe(missing_df, INPUT_DATA)
    return missing_df


def _analysis_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    work = df.copy()
    work["time_to_event"] = pd.to_numeric(work["time_to_event"], errors="coerce")
    work["event"] = pd.to_numeric(work["event"], errors="coerce")
    work = work[work["time_to_event"].notna()].copy()
    work = work[work["time_to_event"] >= 0].copy()
    work = work[work["event"].isin([0, 1])].copy()
    work["event"] = work["event"].astype(int)
    work["treatment_arm"] = work["treatment_arm"].astype(str).str.strip().str.upper()

    cleaning_log = [
        {
            "step": "retain_valid_survival_rows",
            "details": "Rows must have non-negative time_to_event and event in {0, 1}.",
            "rows_after_step": int(len(work)),
        },
        {
            "step": "standardize_treatment_arm_labels",
            "details": "Treatment labels are normalized to {A, B} for later subgroup plotting.",
            "rows_after_step": int(len(work)),
        },
        {
            "step": "preserve_non-required_covariate_missingness",
            "details": "Missing age, stage, and biomarker_x values remain visible; KM overall only requires time and event.",
            "rows_after_step": int(len(work)),
        },
    ]
    return work, cleaning_log


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    SURVIVAL_DIR.mkdir(parents=True, exist_ok=True)

    input_df = _build_input_if_missing()
    analysis_df, cleaning_log = _analysis_dataset(input_df)

    save_dataframe(analysis_df, ANALYSIS_DATA)
    save_dataframe(analysis_df, DATA_DIR / "cleaned.csv")

    event_table = km_event_table(analysis_df, time_col="time_to_event", event_col="event")
    event_table.to_csv(SURVIVAL_DIR / "km_event_table.csv", index=False)

    summaries = plot_km_with_at_risk(
        analysis_df,
        SURVIVAL_DIR / "km_overall.png",
        title="Kaplan–Meier estimate with at-risk table",
        time_unit="days",
    )
    summary_payload = summaries["All"]
    summary_payload["time_grid"] = summaries["time_grid"]["display_times"]

    write_json(QA_DIR / "cleaning_log.json", cleaning_log)
    write_json(SURVIVAL_DIR / "km_overall_summary.json", summary_payload)

    output_files = [
        DATA_DIR / "cleaned.csv",
        QA_DIR / "cleaning_log.json",
        SURVIVAL_DIR / "km_overall.png",
        SURVIVAL_DIR / "km_event_table.csv",
        SURVIVAL_DIR / "km_overall_summary.json",
    ]
    manifest = manifest_payload(
        run_id=RUN_ID,
        input_files={
            "raw_va": RAW_DATA,
            "week02_demo": INPUT_DATA,
            "analysis_dataset": ANALYSIS_DATA,
        },
        output_files=output_files,
        notes={
            "artifact": "Kaplan–Meier + at-risk table",
            "descriptive_only": True,
        },
    )
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
