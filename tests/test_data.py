from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.mlops_project.data import ChestXRayDataset, MyDataset


class TestChestXRayDatasetInitialization:
    """Test dataset initialization and configuration."""

    def test_dataset_initializes_with_valid_path(self, tmp_path: Path) -> None:
        """Test dataset initialization with a valid data directory."""
        split_dir = tmp_path / "train" / "NORMAL"
        split_dir.mkdir(parents=True)

        dataset = ChestXRayDataset(tmp_path, split="train")
        assert isinstance(dataset, Dataset), " Dataset instance not created"
        assert dataset.split == "train", " Split attribute not set correctly"

    def test_dataset_raises_on_missing_split_directory(self, tmp_path: Path) -> None:
        """Test that dataset raises ValueError for missing split directory."""
        with pytest.raises(ValueError, match="does not exist"):
            ChestXRayDataset(tmp_path, split="nonexistent")

    def test_dataset_class_labels_mapping(self, tmp_path: Path) -> None:
        """Test that class labels are correctly mapped."""
        split_dir = tmp_path / "train" / "NORMAL"
        split_dir.mkdir(parents=True)

        dataset = ChestXRayDataset(tmp_path, split="train")
        assert dataset.class_labels == {"NORMAL": 0, "PNEUMONIA": 1}


class TestChestXRayDatasetSchema:
    """Test data schema and shape contracts."""

    @pytest.fixture
    def sample_dataset(self) -> ChestXRayDataset:
        """Create a minimal dataset using a subset of real data."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

        if not data_path.exists():
            pytest.skip("Real dataset not available")

        return ChestXRayDataset(data_path, split="train")

    def test_getitem_returns_tuple(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that __getitem__ returns a tuple of (image, label)."""
        result = sample_dataset[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_image_is_tensor(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that image is returned as a torch.Tensor."""
        image, _ = sample_dataset[0]
        assert isinstance(image, torch.Tensor)

    def test_image_has_correct_shape(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that image has shape [C, H, W] = [3, 224, 224]."""
        image, _ = sample_dataset[0]
        assert len(image.shape) == 3
        assert image.shape == (3, 224, 224), "Image shape is incorrect"

    def test_image_dtype_is_float32(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that image tensor has float32 dtype."""
        image, _ = sample_dataset[0]
        assert image.dtype == torch.float32

    def test_image_values_in_valid_range(
        self, sample_dataset: ChestXRayDataset
    ) -> None:
        """Test that normalized image values are in expected range."""
        image, _ = sample_dataset[0]
        assert torch.isfinite(image).all(), "Image contains NaN or Inf values"
        assert image.min() >= -1.0
        assert image.max() <= 1.0

    def test_label_is_tensor(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that label is returned as a torch.Tensor."""
        _, label = sample_dataset[0]
        assert isinstance(label, torch.Tensor)

    def test_label_is_scalar(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that label is a scalar tensor (shape = ())."""
        _, label = sample_dataset[0]
        assert label.shape == ()

    def test_label_dtype_is_long(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that label tensor has long dtype."""
        _, label = sample_dataset[0]
        assert label.dtype == torch.long

    def test_label_value_in_valid_range(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that label values are valid class indices."""
        _, label = sample_dataset[0]
        assert label.item() in [0, 1]


class TestChestXRayDatasetDeterminism:
    """Test reproducibility and determinism of data loading."""

    @pytest.fixture
    def sample_dataset(self) -> ChestXRayDataset:
        """Create a minimal dataset using a subset of real data."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

        if not data_path.exists():
            pytest.skip("Real dataset not available")

        return ChestXRayDataset(data_path, split="train")

    def test_same_index_returns_consistent_image(
        self, sample_dataset: ChestXRayDataset
    ) -> None:
        """Test that loading the same index twice returns the same image tensor."""
        image_1, _ = sample_dataset[0]
        image_2, _ = sample_dataset[0]
        torch.testing.assert_close(image_1, image_2)

    def test_same_index_returns_consistent_label(
        self, sample_dataset: ChestXRayDataset
    ) -> None:
        """Test that loading the same index twice returns the same label."""
        _, label_1 = sample_dataset[0]
        _, label_2 = sample_dataset[0]
        assert label_1.item() == label_2.item()


class TestChestXRayDatasetLength:
    """Test dataset length properties."""

    @pytest.fixture
    def sample_dataset(self) -> ChestXRayDataset:
        """Create a dataset using real data."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

        if not data_path.exists():
            pytest.skip("Real dataset not available")

        return ChestXRayDataset(data_path, split="train")

    def test_len_returns_total_samples(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that __len__ returns the total number of samples."""
        assert len(sample_dataset) > 0

    def test_image_paths_and_labels_match_length(
        self, sample_dataset: ChestXRayDataset
    ) -> None:
        """Test that image_paths and labels lists match dataset length."""
        assert (
            len(sample_dataset.image_paths)
            == len(sample_dataset.labels)
            == len(sample_dataset)
        )


class TestChestXRayDatasetLabels:
    """Test label handling and class balance."""

    @pytest.fixture
    def sample_dataset(self) -> ChestXRayDataset:
        """Create a dataset using real data."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

        if not data_path.exists():
            pytest.skip("Real dataset not available")

        return ChestXRayDataset(data_path, split="train")

    def test_both_classes_present_in_dataset(
        self, sample_dataset: ChestXRayDataset
    ) -> None:
        """Test that dataset contains both NORMAL and PNEUMONIA samples."""
        unique_labels = set(sample_dataset.labels)
        assert 0 in unique_labels
        assert 1 in unique_labels

    def test_label_assignment_consistency(
        self, sample_dataset: ChestXRayDataset
    ) -> None:
        """Test that labels are valid class indices 0 or 1."""
        for label in sample_dataset.labels:
            assert label in [0, 1]


class TestChestXRayDatasetTransforms:
    """Test data augmentation and transformation pipelines."""

    @pytest.fixture
    def sample_dataset_no_transform(self) -> ChestXRayDataset:
        """Create dataset without transforms using real data."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

        if not data_path.exists():
            pytest.skip("Real dataset not available")

        return ChestXRayDataset(data_path, split="train", transform=None)

    @pytest.fixture
    def sample_dataset_with_transform(self) -> ChestXRayDataset:
        """Create dataset with transforms using real data."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

        if not data_path.exists():
            pytest.skip("Real dataset not available")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return ChestXRayDataset(data_path, split="train", transform=transform)

    def test_default_transform_applied_when_none_provided(
        self,
        sample_dataset_no_transform: ChestXRayDataset,
    ) -> None:
        """Test that default transform is applied when transform=None."""
        image, _ = sample_dataset_no_transform[0]
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32

    def test_custom_transform_applied_correctly(
        self,
        sample_dataset_with_transform: ChestXRayDataset,
    ) -> None:
        """Test that custom transforms are applied correctly."""
        image, _ = sample_dataset_with_transform[0]
        assert image.shape == (3, 224, 224)
        assert torch.isfinite(image).all()


class TestMyDataset:
    """Test MyDataset backwards compatibility alias."""

    def test_mydataset_is_alias_for_chestxraydataset(self) -> None:
        """Test that MyDataset is an alias for ChestXRayDataset."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

        if not data_path.exists():
            pytest.skip("Real dataset not available")

        dataset = MyDataset(data_path, split="train")
        assert isinstance(dataset, ChestXRayDataset)


class TestChestXRayDatasetWithDataLoader:
    """Test dataset integration with PyTorch DataLoader."""

    @pytest.fixture
    def sample_dataset(self) -> ChestXRayDataset:
        """Create a dataset using real data for DataLoader testing."""
        data_path = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

        if not data_path.exists():
            pytest.skip("Real dataset not available")

        return ChestXRayDataset(data_path, split="train")

    def test_dataloader_batch_shapes(self, sample_dataset: ChestXRayDataset) -> None:
        """Test that DataLoader produces correct batch shapes."""
        loader = DataLoader(sample_dataset, batch_size=2)
        images, labels = next(iter(loader))

        assert images.shape == (2, 3, 224, 224)
        assert labels.shape == (2,)

    def test_dataloader_iterates_multiple_batches(
        self, sample_dataset: ChestXRayDataset
    ) -> None:
        """Test that DataLoader can iterate through multiple batches."""
        loader = DataLoader(sample_dataset, batch_size=2, shuffle=False)
        total_samples = 0
        max_batches = 5  # Only test first few batches to keep test fast

        for i, (images, labels) in enumerate(loader):
            total_samples += len(labels)
            if i >= max_batches - 1:
                break

        assert total_samples > 0, "DataLoader did not yield any samples"
        assert total_samples == min(max_batches * 2, len(sample_dataset))

    def test_dataloader_shuffle_changes_order(
        self, sample_dataset: ChestXRayDataset
    ) -> None:
        """Test that shuffling produces different orderings."""
        batch_size = 10
        loader_no_shuffle = DataLoader(
            sample_dataset, batch_size=batch_size, shuffle=False
        )
        loader_shuffle = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True)

        _, labels_no_shuffle = next(iter(loader_no_shuffle))
        _, labels_shuffle = next(iter(loader_shuffle))

        # Shuffling should produce different order (with high probability for 10+ samples)
        assert not torch.equal(labels_no_shuffle, labels_shuffle)
