from pathlib import Path
import hashlib
import json
import pandas as pd

# ---------------------------------
# RATEM Week 1: run validation
# ---------------------------------

HERE = Path(".")
DATA_FILE = HERE / "ratem_demo_survival.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "ratem_demo_survival.csv not found. "
        "Run prepare_week01_data.py first."
    )

# IMPORTANT:
# Read id as string so 001 stays 001
df = pd.read_csv(DATA_FILE, dtype={"id": str})

violations = []

def add_violation(row_idx, field, value, rule, message):
    violations.append({
        "row_index": int(row_idx),
        "row_number_csv": int(row_idx) + 2,  # +1 header +1 zero-index
        "field": field,
        "value": None if pd.isna(value) else str(value),
        "rule": rule,
        "message": message,
    })

# ----------------------------
# 1) Type consistency checks
# ----------------------------
type_fail = False

# time_to_event numeric
time_num = pd.to_numeric(df["time_to_event"], errors="coerce")
bad = df[df["time_to_event"].notna() & time_num.isna()]
for i, row in bad.iterrows():
    type_fail = True
    add_violation(i, "time_to_event", row["time_to_event"], "type_consistency", "Expected numeric")

# event integer-like
event_num = pd.to_numeric(df["event"], errors="coerce")
bad = df[df["event"].notna() & event_num.isna()]
for i, row in bad.iterrows():
    type_fail = True
    add_violation(i, "event", row["event"], "type_consistency", "Expected integer-like numeric")

# age numeric
age_num = pd.to_numeric(df["age"], errors="coerce")
bad = df[df["age"].notna() & age_num.isna()]
for i, row in bad.iterrows():
    type_fail = True
    add_violation(i, "age", row["age"], "type_consistency", "Expected numeric")

# biomarker_x numeric
bio_num = pd.to_numeric(df["biomarker_x"], errors="coerce")
bad = df[df["biomarker_x"].notna() & bio_num.isna()]
for i, row in bad.iterrows():
    type_fail = True
    add_violation(i, "biomarker_x", row["biomarker_x"], "type_consistency", "Expected numeric")

# ----------------------------
# 2) Numeric range checks
# ----------------------------
range_fail = False

# age range 18-100
bad = df[(age_num.notna()) & ((age_num < 18) | (age_num > 100))]
for i, row in bad.iterrows():
    range_fail = True
    add_violation(i, "age", row["age"], "numeric_range", "Allowed range: 18-100")

# biomarker_x range 0-500
bad = df[(bio_num.notna()) & ((bio_num < 0) | (bio_num > 500))]
for i, row in bad.iterrows():
    range_fail = True
    add_violation(i, "biomarker_x", row["biomarker_x"], "numeric_range", "Allowed range: 0-500")

# ----------------------------
# 3) Controlled vocabulary checks
# ----------------------------
vocab_fail = False

# event in {0,1}
bad = df[(event_num.notna()) & (~event_num.isin([0, 1]))]
for i, row in bad.iterrows():
    vocab_fail = True
    add_violation(i, "event", row["event"], "controlled_vocabulary", "Allowed values: [0, 1]")

# treatment_arm in {A,B}
bad = df[df["treatment_arm"].notna() & ~df["treatment_arm"].isin(["A", "B"])]
for i, row in bad.iterrows():
    vocab_fail = True
    add_violation(i, "treatment_arm", row["treatment_arm"], "controlled_vocabulary", "Allowed values: ['A', 'B']")

# stage in {I,II,III,IV}
bad = df[df["stage"].notna() & ~df["stage"].isin(["I", "II", "III", "IV"])]
for i, row in bad.iterrows():
    vocab_fail = True
    add_violation(i, "stage", row["stage"], "controlled_vocabulary", "Allowed values: ['I', 'II', 'III', 'IV']")

# ----------------------------
# 4) ID uniqueness check
# ----------------------------
unique_fail = False

dup = df[df["id"].duplicated(keep=False)]
for i, row in dup.iterrows():
    unique_fail = True
    add_violation(i, "id", row["id"], "id_uniqueness", "ID must be unique")

# ----------------------------
# 5) Temporal logic check
# ----------------------------
temporal_fail = False

bad = df[(time_num.notna()) & (time_num < 0)]
for i, row in bad.iterrows():
    temporal_fail = True
    add_violation(i, "time_to_event", row["time_to_event"], "temporal_logic", "time_to_event must be >= 0")

# ----------------------------
# 6) Missingness report
# ----------------------------
missingness_report = {}
for col in df.columns:
    frac = float(df[col].isna().mean())
    missingness_report[col] = {
        "missing_fraction": round(frac, 4),
        "missing_percent": round(frac * 100, 1),
    }

# ----------------------------
# 7) Check summary
# ----------------------------
checks = [
    {"check_name": "Type Consistency", "status": "FAIL" if type_fail else "PASS"},
    {"check_name": "Numeric Ranges", "status": "FAIL" if range_fail else "PASS"},
    {"check_name": "Controlled Vocabulary", "status": "FAIL" if vocab_fail else "PASS"},
    {"check_name": "ID Uniqueness", "status": "FAIL" if unique_fail else "PASS"},
    {"check_name": "Temporal Logic", "status": "FAIL" if temporal_fail else "PASS"},
    {"check_name": "Missingness Report", "status": "PASS"},
]

# ----------------------------
# 8) Cleaning log
# ----------------------------
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

# ----------------------------
# 9) File hash helper
# ----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ----------------------------
# 10) Main schema report
# ----------------------------
schema_report = {
    "dataset_name": "ratem_demo_survival",
    "input_file": DATA_FILE.name,
    "rows": int(len(df)),
    "columns": list(df.columns),
    "checks": checks,
    "failed_checks": int(sum(c["status"] == "FAIL" for c in checks)),
    "violations_found": int(len(violations)),
    "violations": violations[:50],
}

manifest = {
    "run_id": "week01_validation_local",
    "input_file": DATA_FILE.name,
    "input_sha256": sha256_file(DATA_FILE),
    "output_files": [
        "schema_report.json",
        "missingness_report.json",
        "cleaning_log.json",
        "manifest.json",
    ],
}

# ----------------------------
# 11) Save outputs
# ----------------------------
with open("schema_report.json", "w", encoding="utf-8") as f:
    json.dump(schema_report, f, indent=2)

with open("missingness_report.json", "w", encoding="utf-8") as f:
    json.dump(missingness_report, f, indent=2)

with open("cleaning_log.json", "w", encoding="utf-8") as f:
    json.dump(cleaning_log, f, indent=2)

with open("manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

# ----------------------------
# 12) Print summary
# ----------------------------
print("Validation complete")
print("Saved:")
print("- schema_report.json")
print("- missingness_report.json")
print("- cleaning_log.json")
print("- manifest.json")
print()
print(json.dumps(schema_report, indent=2))
