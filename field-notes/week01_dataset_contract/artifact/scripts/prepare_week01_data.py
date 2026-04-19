from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ratem_demo_utils import build_base_demo_dataset, inject_week01_validation_failures, save_dataframe  # noqa: E402


RAW_FILE = REPO_ROOT / "data" / "raw" / "VA.csv"
DEMO_FILE = REPO_ROOT / "data" / "demo" / "ratem_demo_survival.csv"


def main() -> None:
    base_df = build_base_demo_dataset(RAW_FILE)
    demo_df = inject_week01_validation_failures(base_df)
    save_dataframe(demo_df, DEMO_FILE)

    print(f"Created: {DEMO_FILE.relative_to(REPO_ROOT)}")
    print()
    print("Preview:")
    print(demo_df.head(10))
    print()
    print("Columns:", list(demo_df.columns))
    print("Shape:", demo_df.shape)
    print()
    print("Missingness (%):")
    print((demo_df.isna().mean() * 100).round(1))


if __name__ == "__main__":
    main()
