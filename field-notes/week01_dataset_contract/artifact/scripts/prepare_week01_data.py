from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------
# RATEM Week 1: prepare demo dataset
# ---------------------------------

RNG = np.random.default_rng(42)

HERE = Path(".")
RAW_FILE = HERE / "VA.csv"
DEMO_FILE = HERE / "ratem_demo_survival.csv"

if not RAW_FILE.exists():
    raise FileNotFoundError(
        "VA.csv not found in the current folder. "
        "Place VA.csv in the same folder as this script."
    )

# ----------------------------
# 1) Load raw data
# ----------------------------
df = pd.read_csv(RAW_FILE)

# Drop auto index column if present
first_col = df.columns[0]
if first_col.lower() in {"rownames", "rowname", "x"}:
    df = df.drop(columns=[first_col])

# ----------------------------
# 2) Basic checks on raw columns
# ----------------------------
required_cols = {"stime", "status", "treat", "age", "Karn", "diag.time"}
missing_raw = required_cols - set(df.columns)
if missing_raw:
    raise ValueError(f"VA.csv is missing expected columns: {sorted(missing_raw)}")

# ----------------------------
# 3) Rename columns into RATEM-friendly names
# ----------------------------
df = df.rename(columns={
    "stime": "time_to_event",
    "status": "event",
    "treat": "treatment_arm",
    "age": "age",
})

# ----------------------------
# 4) Standardize event
#    Support either:
#    - numeric 0/1
#    - numeric 1/2
#    - strings like dead/alive/censored
# ----------------------------
raw_event = df["event"]

if raw_event.dtype == object:
    event_str = raw_event.astype(str).str.strip().str.lower()
    mapped = event_str.map({
        "dead": 1,
        "alive": 0,
        "censored": 0,
    })
    if mapped.isna().any():
        numeric_event = pd.to_numeric(raw_event, errors="coerce")
        unique_vals = set(numeric_event.dropna().unique())
        if unique_vals.issubset({0, 1}):
            mapped = numeric_event
        elif unique_vals.issubset({1, 2}):
            mapped = np.where(numeric_event == 2, 1, 0)
        else:
            mapped = numeric_event
    df["event"] = pd.Series(mapped).astype("Int64")
else:
    numeric_event = pd.to_numeric(raw_event, errors="coerce")
    unique_vals = set(numeric_event.dropna().unique())
    if unique_vals.issubset({0, 1}):
        df["event"] = numeric_event.astype("Int64")
    elif unique_vals.issubset({1, 2}):
        df["event"] = pd.Series(np.where(numeric_event == 2, 1, 0)).astype("Int64")
    else:
        df["event"] = numeric_event.astype("Int64")

# ----------------------------
# 5) Standardize treatment arm
#    Support either:
#    - numeric 1/2
#    - strings standard/test
# ----------------------------
raw_treat = df["treatment_arm"]

if raw_treat.dtype == object:
    treat_str = raw_treat.astype(str).str.strip().str.lower()
    df["treatment_arm"] = treat_str.map({
        "standard": "A",
        "test": "B",
        "a": "A",
        "b": "B",
        "1": "A",
        "2": "B",
    })
else:
    treat_num = pd.to_numeric(raw_treat, errors="coerce")
    df["treatment_arm"] = treat_num.map({
        1: "A",
        2: "B",
    })

# ----------------------------
# 6) Create stable string ID
# ----------------------------
df["id"] = [f"{i+1:03d}" for i in range(len(df))]

# ----------------------------
# 7) Build synthetic demo fields
#    These are for RATEM method demo only
# ----------------------------
karn = pd.to_numeric(df["Karn"], errors="coerce")
diag_time = pd.to_numeric(df["diag.time"], errors="coerce")
age = pd.to_numeric(df["age"], errors="coerce")

# Synthetic stage from Karn score
df["stage"] = pd.cut(
    karn.fillna(karn.median()),
    bins=[-np.inf, 40, 60, 80, np.inf],
    labels=["IV", "III", "II", "I"],
).astype(object)

# Synthetic biomarker
df["biomarker_x"] = (
    120
    + 0.6 * age
    + 1.8 * diag_time
    - 0.9 * karn
).clip(0, 500)

# ----------------------------
# 8) Keep only Week 1 contract columns
# ----------------------------
demo = df[
    [
        "id",
        "time_to_event",
        "event",
        "age",
        "treatment_arm",
        "stage",
        "biomarker_x",
    ]
].copy()

# Make these object so we can inject invalid values deliberately
demo["stage"] = demo["stage"].astype(object)
demo["biomarker_x"] = demo["biomarker_x"].astype(object)

# ----------------------------
# 9) Inject deterministic missingness
#    (kept moderate for Week 1 MVP)
# ----------------------------
n = len(demo)

age_idx = RNG.choice(demo.index, size=6, replace=False)
stage_idx = RNG.choice(demo.index, size=12, replace=False)
bio_idx = RNG.choice(demo.index, size=24, replace=False)

demo.loc[age_idx, "age"] = np.nan
demo.loc[stage_idx, "stage"] = np.nan
demo.loc[bio_idx, "biomarker_x"] = np.nan

# ----------------------------
# 10) Inject deliberate validation failures
#     This is intentional for Week 1 narrative
# ----------------------------
demo.loc[1, "time_to_event"] = -12          # temporal logic fail
demo.loc[3, "event"] = 2                    # bad event encoding
demo.loc[4, "age"] = 150                    # out-of-range age
demo.loc[5, "treatment_arm"] = "Arm A"      # invalid controlled vocab
demo.loc[6, "stage"] = "V"                  # invalid controlled vocab
demo.loc[7, "id"] = demo.loc[0, "id"]       # duplicate ID
demo.loc[8, "biomarker_x"] = "oops"         # type mismatch

# Keep explicit column order
demo = demo[
    [
        "id",
        "time_to_event",
        "event",
        "age",
        "treatment_arm",
        "stage",
        "biomarker_x",
    ]
].copy()

# ----------------------------
# 11) Save
# ----------------------------
demo.to_csv(DEMO_FILE, index=False)

# ----------------------------
# 12) Print quick summary
# ----------------------------
print(f"Created: {DEMO_FILE.name}")
print()
print("Preview:")
print(demo.head(10))
print()
print("Columns:", list(demo.columns))
print("Shape:", demo.shape)
print()
print("Missingness (%):")
print((demo.isna().mean() * 100).round(1))
