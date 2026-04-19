from __future__ import annotations

from pathlib import Path
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ratem_demo_utils import (  # noqa: E402
    build_base_demo_dataset,
    inject_policy_aware_missingness,
    manifest_payload,
    missingness_report,
    save_dataframe,
    write_json,
    write_yaml,
)


RUN_ID = "week02_missingness_audit"
RAW_DATA = REPO_ROOT / "data" / "raw" / "VA.csv"
BASE_DATA = REPO_ROOT / "data" / "demo" / "ratem_base_survival.csv"
MISSINGNESS_DATA = REPO_ROOT / "data" / "demo" / "ratem_demo_missingness.csv"
POLICY_FILE = Path(__file__).resolve().parent / "missingness_policy.yml"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / RUN_ID
QA_DIR = OUTPUT_DIR / "qa"
EDA_DIR = OUTPUT_DIR / "eda"


def _co_missingness_report(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    pairwise: dict[str, dict[str, int]] = {}
    for left in columns:
        pairwise[left] = {}
        for right in columns:
            pairwise[left][right] = int((df[left].isna() & df[right].isna()).sum())
    return pairwise


def _tracking_report(df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    report: dict[str, dict] = {
        "by_treatment_arm": {},
        "by_event": {},
        "by_followup_band": {},
    }

    work = df.copy()
    work["event"] = pd.to_numeric(work["event"], errors="coerce")
    work["time_to_event"] = pd.to_numeric(work["time_to_event"], errors="coerce")
    work["followup_band"] = pd.qcut(
        work["time_to_event"],
        q=4,
        labels=["Q1_shortest", "Q2", "Q3", "Q4_longest"],
        duplicates="drop",
    )

    for column in columns:
        report["by_treatment_arm"][column] = {
            str(group): round(float(subframe[column].isna().mean() * 100), 1)
            for group, subframe in work.groupby("treatment_arm", dropna=False)
        }
        report["by_event"][column] = {
            f"event_{int(group)}": round(float(subframe[column].isna().mean() * 100), 1)
            for group, subframe in work.groupby("event", dropna=False)
        }
        report["by_followup_band"][column] = {
            str(group): round(float(subframe[column].isna().mean() * 100), 1)
            for group, subframe in work.groupby("followup_band", dropna=False, observed=False)
        }

    return report


def _cleaning_log(df: pd.DataFrame) -> list[dict]:
    full_report = missingness_report(df)
    policy_notes = {
        "age": "occasional entry gaps",
        "stage": "reporting/site variation",
        "biomarker_x": "test not ordered for everyone",
    }
    actions = {
        "age": "flag",
        "stage": "review",
        "biomarker_x": "explicit_policy_required",
    }

    rows = []
    for field in ["age", "stage", "biomarker_x"]:
        rows.append(
            {
                "field": field,
                "missing_percent": full_report[field]["missing_percent"],
                "note": policy_notes[field],
                "action": actions[field],
            }
        )
    return rows


def _plot_missingness_heatmap(df: pd.DataFrame, columns: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = df.sort_values(["treatment_arm", "time_to_event"]).reset_index(drop=True)
    matrix = ordered[columns].isna().astype(int).to_numpy().T

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greys")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    ax.set_xticks([])
    ax.set_xlabel("Rows (ordered by treatment_arm, then follow-up time)")
    ax.set_title("Policy-aware missingness heatmap")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    base_df = build_base_demo_dataset(RAW_DATA)
    missing_df = inject_policy_aware_missingness(base_df)

    save_dataframe(base_df, BASE_DATA)
    save_dataframe(missing_df, MISSINGNESS_DATA)

    if POLICY_FILE.exists():
        with POLICY_FILE.open("r", encoding="utf-8") as handle:
            policy = yaml.safe_load(handle)
    else:
        policy = {
            "missingness": {
                "fields": ["age", "stage", "biomarker_x"],
                "report": {
                    "by_variable": True,
                    "co_missingness": True,
                    "track_with_covariates": True,
                    "track_with_outcome_followup": True,
                },
                "silent_row_drop": False,
                "actions_must_be_logged": True,
            }
        }
        write_yaml(POLICY_FILE, policy)

    tracked_columns = policy["missingness"]["fields"]
    variable_report = {key: value for key, value in missingness_report(missing_df).items() if key in tracked_columns}
    pairwise_report = _co_missingness_report(missing_df, tracked_columns)
    tracking = _tracking_report(missing_df, tracked_columns)
    cleaning_log = _cleaning_log(missing_df)

    heatmap_path = EDA_DIR / "missingness_heatmap.png"
    _plot_missingness_heatmap(missing_df, tracked_columns, heatmap_path)

    write_json(QA_DIR / "missingness_report.json", variable_report)
    write_json(QA_DIR / "co_missingness_report.json", pairwise_report)
    write_json(QA_DIR / "missingness_tracking.json", tracking)
    write_json(QA_DIR / "cleaning_log.json", cleaning_log)

    output_files = [
        QA_DIR / "missingness_report.json",
        QA_DIR / "co_missingness_report.json",
        QA_DIR / "missingness_tracking.json",
        QA_DIR / "cleaning_log.json",
        EDA_DIR / "missingness_heatmap.png",
    ]
    manifest = manifest_payload(
        run_id=RUN_ID,
        input_files={
            "raw_va": RAW_DATA,
            "demo_missingness": MISSINGNESS_DATA,
            "policy": POLICY_FILE,
        },
        output_files=output_files,
        notes={
            "tracked_fields": tracked_columns,
            "silent_row_drop": policy["missingness"]["silent_row_drop"],
        },
    )
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    summary = {
        "run_id": RUN_ID,
        "dataset_rows": int(len(missing_df)),
        "tracked_fields": tracked_columns,
        "missingness_percent": {
            field: variable_report[field]["missing_percent"]
            for field in tracked_columns
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
