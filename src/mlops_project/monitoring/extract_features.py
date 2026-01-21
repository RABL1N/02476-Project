from pathlib import Path
import pandas as pd

from mlops_project.data import ChestXRayDataset


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def extract_features(dataset: ChestXRayDataset) -> pd.DataFrame:
    records = []

    for image, label in dataset:
        img = image.numpy()

        records.append(
            {
                "mean_intensity": img.mean(),
                "std_intensity": img.std(),
                "min_intensity": img.min(),
                "max_intensity": img.max(),
                "label": label.item(),
            }
        )

    return pd.DataFrame(records)


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    data_path = Path("data/raw/chest_xray")

    train_ds = ChestXRayDataset(data_path, split="train")
    test_ds = ChestXRayDataset(data_path, split="test")

    reference_df = extract_features(train_ds)
    current_df = extract_features(test_ds)

    reference_df.to_csv(ARTIFACTS_DIR / "reference_features.csv", index=False)
    current_df.to_csv(ARTIFACTS_DIR / "current_features.csv", index=False)

    print("Feature extraction complete.")
    print(f"Saved artifacts to: {ARTIFACTS_DIR.resolve()}")
    print(f"Train samples: {len(reference_df)}")
    print(f"Test samples: {len(current_df)}")


if __name__ == "__main__":
    main()
