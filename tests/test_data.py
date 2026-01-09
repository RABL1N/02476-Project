from pathlib import Path

import torch
from torch.utils.data import Dataset

from mlops_project.data import ChestXRayDataset, MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(Path("data/raw"), split="train")
    assert isinstance(dataset, Dataset)


def test_chest_xray_dataset_initialization():
    """Test that ChestXRayDataset can be initialized."""
    dataset = ChestXRayDataset(Path("data/raw"), split="train")
    assert isinstance(dataset, ChestXRayDataset)
    assert len(dataset) > 0


def test_chest_xray_dataset_length():
    """Test that dataset returns correct length."""
    train_dataset = ChestXRayDataset(Path("data/raw"), split="train")
    val_dataset = ChestXRayDataset(Path("data/raw"), split="val")
    test_dataset = ChestXRayDataset(Path("data/raw"), split="test")
    
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(test_dataset) > 0


def test_chest_xray_dataset_getitem():
    """Test that __getitem__ returns correct format."""
    dataset = ChestXRayDataset(Path("data/raw"), split="train")
    
    # Get first item
    image, label = dataset[0]
    
    # Check types
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    
    # Check image shape (should be [C, H, W] after transform)
    assert len(image.shape) == 3
    assert image.shape[0] == 3  # RGB channels
    assert image.shape[1] == 224  # Height
    assert image.shape[2] == 224  # Width
    
    # Check label is a scalar tensor
    assert label.shape == ()
    assert label.item() in [0, 1]  # NORMAL=0, PNEUMONIA=1


def test_chest_xray_dataset_labels():
    """Test that labels are correct."""
    dataset = ChestXRayDataset(Path("data/raw"), split="train")
    
    # Check that we have both classes (sample across the entire dataset)
    # Sample from different parts to ensure we get both classes
    indices = [0, len(dataset) // 4, len(dataset) // 2, 3 * len(dataset) // 4, len(dataset) - 1]
    labels = [dataset[i][1].item() for i in indices]
    unique_labels = set(labels)
    assert 0 in unique_labels  # NORMAL
    assert 1 in unique_labels  # PNEUMONIA
    
    # Also check all labels to ensure both classes exist
    all_labels = set(dataset.labels)
    assert 0 in all_labels  # NORMAL
    assert 1 in all_labels  # PNEUMONIA


def test_chest_xray_dataset_all_splits():
    """Test that all splits work correctly."""
    for split in ["train", "val", "test"]:
        dataset = ChestXRayDataset(Path("data/raw"), split=split)
        assert len(dataset) > 0
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
