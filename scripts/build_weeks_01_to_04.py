from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

SCRIPTS = [
    "field-notes/week01_dataset_contract/artifact/scripts/prepare_week01_data.py",
    "field-notes/week01_dataset_contract/artifact/scripts/run_validation.py",
    "field-notes/week02_missingness_policy/artifact/scripts/build_week02_missingness_artifacts.py",
    "field-notes/week03_kaplan_meier_baseline/artifact/scripts/build_week03_km_artifacts.py",
    "field-notes/week04_stratified_km/artifact/scripts/build_week04_stratified_km_artifacts.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        subprocess.run([PYTHON, script], cwd=REPO_ROOT, check=True)

    print("\nWeeks 1–4 artifacts regenerated successfully.")


if __name__ == "__main__":
    main()
