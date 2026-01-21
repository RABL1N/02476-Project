from pathlib import Path
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    reference_df = pd.read_csv(ARTIFACTS_DIR / "reference_features.csv")
    current_df = pd.read_csv(ARTIFACTS_DIR / "current_features.csv")

    # (Optional) label isn't part of feature drift checking
    if "label" in reference_df.columns:
        reference_df = reference_df.drop(columns=["label"])
    if "label" in current_df.columns:
        current_df = current_df.drop(columns=["label"])

    # In Evidently 0.7.x: use Presets inside Report([...])
    report = Report(
        [
            DataDriftPreset(),
            # Optional but nice for the DTU guide: adds data quality summary
            DataSummaryPreset(),
        ],
        include_tests=True,  # optional: gives pass/fail style tests too
    )

    # IMPORTANT: first arg = current data, second arg = reference data
    my_eval = report.run(current_df, reference_df)

    out_path = ARTIFACTS_DIR / "data_drift_report.html"
    my_eval.save_html(str(out_path))

    print(f"Saved drift report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
