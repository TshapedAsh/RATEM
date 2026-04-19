from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


DEFAULT_SEED = 2026


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_yaml(path: Path, payload: object) -> None:
    """Write YAML with stable formatting."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def sha256_file(path: Path) -> str:
    """Compute the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _drop_auto_index_column(df: pd.DataFrame) -> pd.DataFrame:
    first_col = df.columns[0]
    if first_col.lower() in {"rownames", "rowname", "x"}:
        return df.drop(columns=[first_col])
    return df


def _standardize_event(raw_event: pd.Series) -> pd.Series:
    numeric_event = pd.to_numeric(raw_event, errors="coerce")
    unique_vals = set(numeric_event.dropna().unique())
    if unique_vals.issubset({0, 1}):
        mapped = numeric_event
    elif unique_vals.issubset({1, 2}):
        mapped = pd.Series(np.where(numeric_event == 2, 1, 0), index=raw_event.index)
    else:
        mapped = numeric_event
    return pd.Series(mapped, index=raw_event.index).astype("Int64")


def _standardize_treatment(raw_treatment: pd.Series) -> pd.Series:
    if raw_treatment.dtype == object:
        treatment = raw_treatment.astype(str).str.strip().str.lower().map(
            {
                "a": "A",
                "b": "B",
                "1": "A",
                "2": "B",
                "standard": "A",
                "test": "B",
                "arm a": "A",
                "arm b": "B",
            }
        )
        return treatment

    numeric_treatment = pd.to_numeric(raw_treatment, errors="coerce")
    return numeric_treatment.map({1: "A", 2: "B"})


def load_va_raw(raw_path: Path) -> pd.DataFrame:
    """Load the Veterans' Administration dataset used for the demo build."""
    df = pd.read_csv(raw_path)
    df = _drop_auto_index_column(df)

    required_cols = {"stime", "status", "treat", "age", "Karn", "diag.time"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Raw VA dataset is missing expected columns: {missing_cols}")
    return df


def build_base_demo_dataset(raw_path: Path) -> pd.DataFrame:
    """Create the canonical RATEM demo dataset used across Weeks 2–4.

    Notes
    -----
    - The base event-time information comes from the VA dataset.
    - `stage` and `biomarker_x` are synthetic demonstration fields so the public
      field notes can illustrate validation and missingness behavior.
    """
    raw = load_va_raw(raw_path).copy()

    raw = raw.rename(
        columns={
            "stime": "time_to_event",
            "status": "event",
            "treat": "treatment_arm",
        }
    )

    raw["event"] = _standardize_event(raw["event"])
    raw["treatment_arm"] = _standardize_treatment(raw["treatment_arm"])
    raw["age"] = pd.to_numeric(raw["age"], errors="coerce")
    raw["time_to_event"] = pd.to_numeric(raw["time_to_event"], errors="coerce")

    raw["id"] = [f"{idx + 1:03d}" for idx in range(len(raw))]

    karn = pd.to_numeric(raw["Karn"], errors="coerce")
    diag_time = pd.to_numeric(raw["diag.time"], errors="coerce")
    age = pd.to_numeric(raw["age"], errors="coerce")

    raw["stage"] = pd.cut(
        karn.fillna(karn.median()),
        bins=[-np.inf, 40, 60, 80, np.inf],
        labels=["IV", "III", "II", "I"],
    ).astype("object")

    biomarker = (120 + 0.6 * age + 1.8 * diag_time - 0.9 * karn).clip(0, 500)
    raw["biomarker_x"] = biomarker.round(1)

    return raw[
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


def _weighted_sample(index: pd.Index, size: int, weights: Iterable[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = np.asarray(list(weights), dtype=float)
    if probabilities.size != len(index):
        raise ValueError("Weights must have the same length as the index.")
    if (probabilities < 0).any():
        raise ValueError("Weights must be non-negative.")
    if probabilities.sum() == 0:
        probabilities = np.ones_like(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return rng.choice(np.asarray(index), size=size, replace=False, p=probabilities)




def inject_week01_validation_failures(base_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create the intentionally flawed Week 1 validation demo dataset."""
    rng = np.random.default_rng(seed)
    demo = base_df.copy()
    demo["stage"] = demo["stage"].astype(object)
    demo["biomarker_x"] = demo["biomarker_x"].astype(object)

    age_idx = rng.choice(demo.index, size=6, replace=False)
    stage_idx = rng.choice(demo.index, size=12, replace=False)
    bio_idx = rng.choice(demo.index, size=24, replace=False)

    demo.loc[age_idx, "age"] = np.nan
    demo.loc[stage_idx, "stage"] = np.nan
    demo.loc[bio_idx, "biomarker_x"] = np.nan

    demo.loc[1, "time_to_event"] = -12
    demo.loc[3, "event"] = 2
    demo.loc[4, "age"] = 150
    demo.loc[5, "treatment_arm"] = "Arm A"
    demo.loc[6, "stage"] = "V"
    demo.loc[7, "id"] = demo.loc[0, "id"]
    demo.loc[8, "biomarker_x"] = "oops"
    return demo
def inject_policy_aware_missingness(base_df: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Inject deterministic, structured missingness for the Week 2 artifact.

    The target rates are approximately:
    - age: 5%
    - stage: 12%
    - biomarker_x: 45%
    """
    df = base_df.copy()

    original_age = df["age"].copy()
    original_event = pd.to_numeric(df["event"], errors="coerce")
    original_time = pd.to_numeric(df["time_to_event"], errors="coerce")
    original_treatment = df["treatment_arm"].astype(str)

    age_count = 7
    stage_count = 17
    biomarker_count = 62

    # Mildly structured age missingness.
    age_weights = 1 + 0.5 * (original_treatment == "B") + 0.25 * (original_event == 0)
    age_idx = _weighted_sample(df.index, age_count, age_weights, seed=seed)

    # Stage missingness is more frequent in arm B and longer follow-up rows.
    stage_weights = 1 + 1.25 * (original_treatment == "B") + 0.5 * (original_time > original_time.median())
    stage_idx = _weighted_sample(df.index, stage_count, stage_weights, seed=seed + 1)

    # Biomarker missingness is intentionally structured to mimic a test that is
    # not ordered uniformly across the cohort.
    biomarker_weights = (
        1
        + 1.5 * (original_treatment == "B")
        + 0.9 * (original_age >= 60)
        + 0.5 * (original_event == 1)
    )
    biomarker_idx = _weighted_sample(df.index, biomarker_count, biomarker_weights, seed=seed + 2)

    df.loc[age_idx, "age"] = np.nan
    df.loc[stage_idx, "stage"] = np.nan
    df.loc[biomarker_idx, "biomarker_x"] = np.nan

    return df


def missingness_report(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return variable-wise missingness fractions and percents."""
    report: dict[str, dict[str, float]] = {}
    for column in df.columns:
        fraction = float(df[column].isna().mean())
        report[column] = {
            "missing_fraction": round(fraction, 4),
            "missing_percent": round(fraction * 100, 1),
        }
    return report


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe as CSV with parent directory creation."""
    ensure_parent(path)
    df.to_csv(path, index=False)


def manifest_payload(run_id: str, input_files: dict[str, Path], output_files: list[Path], notes: dict | None = None) -> dict:
    """Build a simple run manifest payload."""
    payload = {
        "run_id": run_id,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in input_files.items()
        },
        "outputs": [str(path) for path in output_files],
    }
    if notes:
        payload["notes"] = notes
    return payload
