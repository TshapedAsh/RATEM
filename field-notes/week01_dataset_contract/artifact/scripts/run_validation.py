from __future__ import annotations

from pathlib import Path
import sys
import json

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ratem_demo_utils import manifest_payload, write_json  # noqa: E402


DATA_FILE = REPO_ROOT / "data" / "demo" / "ratem_demo_survival.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


violations: list[dict] = []


def add_violation(row_idx: int, field: str, value: object, rule: str, message: str) -> None:
    violations.append(
        {
            "row_index": int(row_idx),
            "row_number_csv": int(row_idx) + 2,
            "field": field,
            "value": None if pd.isna(value) else str(value),
            "rule": rule,
            "message": message,
        }
    )


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "ratem_demo_survival.csv not found. Run prepare_week01_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE, dtype={"id": str})

    # 1) Type checks
    type_fail = False
    time_num = pd.to_numeric(df["time_to_event"], errors="coerce")
    event_num = pd.to_numeric(df["event"], errors="coerce")
    age_num = pd.to_numeric(df["age"], errors="coerce")
    bio_num = pd.to_numeric(df["biomarker_x"], errors="coerce")

    for field, numeric_series, message in [
        ("time_to_event", time_num, "Expected numeric"),
        ("event", event_num, "Expected integer-like numeric"),
        ("age", age_num, "Expected numeric"),
        ("biomarker_x", bio_num, "Expected numeric"),
    ]:
        bad = df[df[field].notna() & numeric_series.isna()]
        for row_index, row in bad.iterrows():
            type_fail = True
            add_violation(row_index, field, row[field], "type_consistency", message)

    # 2) Range checks
    range_fail = False
    bad = df[(age_num.notna()) & ((age_num < 18) | (age_num > 100))]
    for row_index, row in bad.iterrows():
        range_fail = True
        add_violation(row_index, "age", row["age"], "numeric_range", "Allowed range: 18-100")

    bad = df[(bio_num.notna()) & ((bio_num < 0) | (bio_num > 500))]
    for row_index, row in bad.iterrows():
        range_fail = True
        add_violation(row_index, "biomarker_x", row["biomarker_x"], "numeric_range", "Allowed range: 0-500")

    # 3) Controlled vocabulary
    vocab_fail = False
    bad = df[(event_num.notna()) & (~event_num.isin([0, 1]))]
    for row_index, row in bad.iterrows():
        vocab_fail = True
        add_violation(row_index, "event", row["event"], "controlled_vocabulary", "Allowed values: [0, 1]")

    bad = df[df["treatment_arm"].notna() & ~df["treatment_arm"].isin(["A", "B"])]
    for row_index, row in bad.iterrows():
        vocab_fail = True
        add_violation(row_index, "treatment_arm", row["treatment_arm"], "controlled_vocabulary", "Allowed values: ['A', 'B']")

    bad = df[df["stage"].notna() & ~df["stage"].isin(["I", "II", "III", "IV"])]
    for row_index, row in bad.iterrows():
        vocab_fail = True
        add_violation(row_index, "stage", row["stage"], "controlled_vocabulary", "Allowed values: ['I', 'II', 'III', 'IV']")

    # 4) ID uniqueness
    unique_fail = False
    dup = df[df["id"].duplicated(keep=False)]
    for row_index, row in dup.iterrows():
        unique_fail = True
        add_violation(row_index, "id", row["id"], "id_uniqueness", "ID must be unique")

    # 5) Temporal logic
    temporal_fail = False
    bad = df[(time_num.notna()) & (time_num < 0)]
    for row_index, row in bad.iterrows():
        temporal_fail = True
        add_violation(row_index, "time_to_event", row["time_to_event"], "temporal_logic", "time_to_event must be >= 0")

    # 6) Missingness
    missingness = {}
    for column in df.columns:
        frac = float(df[column].isna().mean())
        missingness[column] = {
            "missing_fraction": round(frac, 4),
            "missing_percent": round(frac * 100, 1),
        }

    checks = [
        {"check_name": "Type Consistency", "status": "FAIL" if type_fail else "PASS"},
        {"check_name": "Numeric Ranges", "status": "FAIL" if range_fail else "PASS"},
        {"check_name": "Controlled Vocabulary", "status": "FAIL" if vocab_fail else "PASS"},
        {"check_name": "ID Uniqueness", "status": "FAIL" if unique_fail else "PASS"},
        {"check_name": "Temporal Logic", "status": "FAIL" if temporal_fail else "PASS"},
        {"check_name": "Missingness Report", "status": "PASS"},
    ]

    cleaning_log = [
        {
            "rule": "fix_negative_time",
            "action": "Review and set invalid time_to_event to NaN only with provenance note",
            "applied": False,
        },
        {
            "rule": "review_event_encoding",
            "action": "Enforce event in {0,1}",
            "applied": False,
        },
        {
            "rule": "review_out_of_range_age",
            "action": "Flag or set invalid age to NaN according to policy",
            "applied": False,
        },
        {
            "rule": "normalize_controlled_vocabulary",
            "action": "Map treatment_arm or stage only with documented rules",
            "applied": False,
        },
        {
            "rule": "resolve_duplicate_ids",
            "action": "Review duplicate IDs before deduplication",
            "applied": False,
        },
        {
            "rule": "fix_type_mismatches",
            "action": "Review non-numeric biomarker entries before coercion",
            "applied": False,
        },
    ]

    schema_report = {
        "dataset_name": "ratem_demo_survival",
        "input_file": str(DATA_FILE.relative_to(REPO_ROOT)),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "checks": checks,
        "failed_checks": int(sum(item["status"] == "FAIL" for item in checks)),
        "violations_found": int(len(violations)),
        "violations": violations[:50],
    }

    write_json(OUTPUT_DIR / "schema_report.json", schema_report)
    write_json(OUTPUT_DIR / "missingness_report.json", missingness)
    write_json(OUTPUT_DIR / "cleaning_log.json", cleaning_log)

    manifest = manifest_payload(
        run_id="week01_validation_local",
        input_files={"validation_demo": DATA_FILE},
        output_files=[
            OUTPUT_DIR / "schema_report.json",
            OUTPUT_DIR / "missingness_report.json",
            OUTPUT_DIR / "cleaning_log.json",
        ],
        notes={"week": 1, "artifact": "Validation"},
    )
    write_json(OUTPUT_DIR / "manifest.json", manifest)

    print("Validation complete")
    print(json.dumps(schema_report, indent=2))


if __name__ == "__main__":
    main()
