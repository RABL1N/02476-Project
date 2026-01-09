"""Simple script to test the dataloader."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.mlops_project.data import ChestXRayDataset

if __name__ == "__main__":
    print("Testing ChestXRayDataset...")
    print("=" * 50)
    
    # Test all splits
    for split in ["train", "val", "test"]:
        print(f"\nTesting {split} split:")
        dataset = ChestXRayDataset(Path("data/raw"), split=split)
        print(f"  Dataset size: {len(dataset)}")
        
        # Test getting an item
        image, label = dataset[0]
        print(f"  First image shape: {image.shape}")
        print(f"  First label: {label.item()} ({'NORMAL' if label.item() == 0 else 'PNEUMONIA'})")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Test with DataLoader
    print("\n" + "=" * 50)
    print("\nTesting with DataLoader:")
    train_dataset = ChestXRayDataset(Path("data/raw"), split="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"  Batch images shape: {images.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Batch labels: {labels.tolist()}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
