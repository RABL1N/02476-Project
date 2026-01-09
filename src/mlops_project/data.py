from pathlib import Path
from typing import Optional, Tuple

import typer
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXRayDataset(Dataset):
    """Dataset for Chest X-Ray Pneumonia images."""

    def __init__(
        self,
        data_path: Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_path: Path to the root of the dataset (contains train/test/val folders)
            split: Dataset split to use ('train', 'val', or 'test')
            transform: Optional torchvision transforms to apply to images
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform

        # Class labels
        self.class_labels = {"NORMAL": 0, "PNEUMONIA": 1}

        # Load image paths and labels
        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        split_dir = self.data_path / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")

        for label_name, label_idx in self.class_labels.items():
            class_dir = split_dir / label_name
            if class_dir.exists():
                image_files = sorted([
                    f for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ])
                self.image_paths.extend(image_files)
                self.labels.extend([label_idx] * len(image_files))

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a given sample from the dataset.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Tuple of (image tensor, label tensor)
        """
        img_path = self.image_paths[index]
        label = self.labels[index]

        # Load image and convert to RGB (handles grayscale and RGB images)
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default transform: resize and convert to tensor
            default_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
            image = default_transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder.

        Args:
            output_folder: Path to save preprocessed data
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        # For this dataset, preprocessing is minimal as images are already in good format
        # This method can be extended for additional preprocessing if needed
        print(f"Dataset contains {len(self)} samples in {self.split} split")


class MyDataset(ChestXRayDataset):
    """Alias for ChestXRayDataset for backward compatibility."""

    pass


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Preprocess the dataset."""
    print("Preprocessing data...")
    dataset = ChestXRayDataset(data_path, split="train")
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
