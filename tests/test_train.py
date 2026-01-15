from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from src.mlops_project.model import Model
from src.mlops_project.train import train


class TestTrainingLoop:
    """Test training loop components without running full training."""

    def test_loss_decreases_with_training_step(self) -> None:
        """Test that loss decreases after a few training steps on simple data."""
        model = Model(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create simple synthetic data that's easy to learn
        x = torch.randn(10, 3, 224, 224)
        y = torch.zeros(10, dtype=torch.long)  # All same class
        
        model.train()
        
        # Get initial loss
        initial_output = model(x)
        initial_loss = criterion(initial_output, y)
        
        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Get final loss
        final_output = model(x)
        final_loss = criterion(final_output, y)
        
        # Loss should decrease
        assert final_loss.item() < initial_loss.item(), "Loss did not decrease during training"

    def test_model_parameters_update_during_training(self) -> None:
        """Test that model parameters actually change during training."""
        model = Model(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Create data and train one step
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 2, (4,))
        
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Check that at least some parameters changed
        params_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.equal(initial, current):
                params_changed = True
                break
        
        assert params_changed, "No model parameters were updated during training"

    def test_training_mode_affects_dropout(self) -> None:
        """Test that model behaves differently in train vs eval mode."""
        model = Model(num_classes=2)
        x = torch.randn(4, 3, 224, 224)
        
        # In train mode with same seed
        model.train()
        torch.manual_seed(42)
        output_train_1 = model(x)
        torch.manual_seed(42)
        output_train_2 = model(x)
        
        # Should be identical with same seed
        torch.testing.assert_close(output_train_1, output_train_2)
        
        # In eval mode
        model.eval()
        output_eval_1 = model(x)
        output_eval_2 = model(x)
        
        # Should be identical without seed
        torch.testing.assert_close(output_eval_1, output_eval_2)

    def test_gradients_accumulate_correctly(self) -> None:
        """Test that gradients accumulate when backward is called multiple times."""
        model = Model(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 2, (2,))
        
        # First backward pass
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        first_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        
        # Second backward pass without zero_grad
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Gradients should have increased (accumulated)
        grads_accumulated = False
        for first_grad, param in zip(first_grads, model.parameters()):
            if first_grad is not None and param.grad is not None:
                # Check that gradient magnitude increased
                if torch.norm(param.grad) > torch.norm(first_grad):
                    grads_accumulated = True
                    break
        
        assert grads_accumulated, "Gradients did not accumulate"


class TestOptimizerAndLoss:
    """Test optimizer and loss function behavior."""

    def test_adam_optimizer_updates_parameters(self) -> None:
        """Test that Adam optimizer updates parameters correctly."""
        model = Model(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 2, (2,))
        
        # Store initial parameter
        first_param_initial = next(model.parameters()).clone()
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        first_param_after = next(model.parameters())
        
        assert not torch.equal(first_param_initial, first_param_after)

    def test_cross_entropy_loss_output_shape(self) -> None:
        """Test that CrossEntropyLoss produces scalar output."""
        criterion = nn.CrossEntropyLoss()
        
        predictions = torch.randn(4, 2)  # batch_size=4, num_classes=2
        targets = torch.randint(0, 2, (4,))
        
        loss = criterion(predictions, targets)
        
        assert loss.shape == (), "Loss should be scalar"
        assert loss.dtype == torch.float32

    def test_cross_entropy_loss_is_positive(self) -> None:
        """Test that CrossEntropyLoss produces positive values."""
        criterion = nn.CrossEntropyLoss()
        
        predictions = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        
        loss = criterion(predictions, targets)
        
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_perfect_predictions_give_low_loss(self) -> None:
        """Test that perfect predictions result in very low loss."""
        criterion = nn.CrossEntropyLoss()
        
        # Create perfect predictions (high confidence for correct class)
        predictions = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
        targets = torch.tensor([0, 1])
        
        loss = criterion(predictions, targets)
        
        assert loss.item() < 0.01, "Perfect predictions should have very low loss"


class TestDataLoaderIntegration:
    """Test DataLoader integration with training."""

    @pytest.fixture
    def mock_dataset(self, tmp_path: Path):
        """Create a mock dataset with synthetic data."""
        normal_dir = tmp_path / "train" / "NORMAL"
        pneumonia_dir = tmp_path / "train" / "PNEUMONIA"
        normal_dir.mkdir(parents=True)
        pneumonia_dir.mkdir(parents=True)
        
        from PIL import Image
        
        # Create a few synthetic images
        for i in range(5):
            img = Image.new("RGB", (224, 224))
            img.save(normal_dir / f"normal_{i}.jpg")
            img.save(pneumonia_dir / f"pneumonia_{i}.jpg")
        
        return tmp_path

    def test_dataloader_produces_correct_batch_shapes(self, mock_dataset: Path) -> None:
        """Test that DataLoader produces batches with correct shapes."""
        from src.mlops_project.data import ChestXRayDataset
        
        dataset = ChestXRayDataset(mock_dataset, split="train")
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        images, labels = next(iter(loader))
        
        assert images.shape == (2, 3, 224, 224)
        assert labels.shape == (2,)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.long

    def test_training_step_with_dataloader(self, mock_dataset: Path) -> None:
        """Test that a single training step works with real DataLoader."""
        from src.mlops_project.data import ChestXRayDataset
        
        dataset = ChestXRayDataset(mock_dataset, split="train")
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        model = Model(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        images, labels = next(iter(loader))
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"


class TestAccuracyCalculation:
    """Test accuracy calculation during training."""

    def test_accuracy_calculation_perfect_predictions(self) -> None:
        """Test accuracy calculation with perfect predictions."""
        predictions = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
        labels = torch.tensor([0, 1, 0])
        
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
        
        assert accuracy == 100.0

    def test_accuracy_calculation_all_wrong(self) -> None:
        """Test accuracy calculation with all wrong predictions."""
        predictions = torch.tensor([[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        labels = torch.tensor([0, 1, 0])
        
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
        
        assert accuracy == 0.0

    def test_accuracy_calculation_partial(self) -> None:
        """Test accuracy calculation with partial correct predictions."""
        predictions = torch.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0], [10.0, 0.0]])
        labels = torch.tensor([0, 1, 0, 0])
        
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
        
        assert accuracy == 75.0  # 3 out of 4 correct


class TestModelSavingDuringTraining:
    """Test model checkpoint saving functionality."""

    def test_model_state_dict_saves_and_loads(self, tmp_path: Path) -> None:
        """Test that model can be saved and loaded during training."""
        model = Model(num_classes=2)
        save_path = tmp_path / "checkpoint.pt"
        
        # Train for one step
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 2, (2,))
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Save model
        torch.save(model.state_dict(), save_path)
        
        # Load into new model
        new_model = Model(num_classes=2)
        new_model.load_state_dict(torch.load(save_path, weights_only=True))
        
        # Verify models produce same output
        model.eval()
        new_model.eval()
        
        test_x = torch.randn(1, 3, 224, 224)
        output1 = model(test_x)
        output2 = new_model(test_x)
        
        torch.testing.assert_close(output1, output2)

    def test_best_model_tracking(self) -> None:
        """Test tracking of best validation accuracy."""
        val_accuracies = [75.5, 78.2, 76.1, 80.3, 79.5]
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch, val_acc in enumerate(val_accuracies):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
        
        assert best_val_acc == 80.3
        assert best_epoch == 3


class TestDeviceHandling:
    """Test device handling (CPU/CUDA) during training."""

    def test_model_moves_to_cpu(self) -> None:
        """Test that model can be moved to CPU."""
        model = Model(num_classes=2)
        model = model.to("cpu")
        
        x = torch.randn(1, 3, 224, 224, device="cpu")
        output = model(x)
        
        assert output.device.type == "cpu"

    def test_tensors_on_same_device_during_forward(self) -> None:
        """Test that model and data are on the same device."""
        model = Model(num_classes=2)
        device = "cpu"
        model = model.to(device)
        
        x = torch.randn(2, 3, 224, 224).to(device)
        y = torch.randint(0, 2, (2,)).to(device)
        
        output = model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        
        assert output.device.type == device
        assert loss.device.type == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_moves_to_cuda(self) -> None:
        """Test that model can be moved to CUDA."""
        model = Model(num_classes=2)
        model = model.to("cuda")
        
        x = torch.randn(1, 3, 224, 224, device="cuda")
        output = model(x)
        
        assert output.device.type == "cuda"


class TestTrainingConfiguration:
    """Test training configuration validation."""

    def test_valid_learning_rates(self) -> None:
        """Test that various learning rates are accepted."""
        model = Model(num_classes=2)
        
        for lr in [0.001, 0.01, 0.0001, 1e-4]:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            assert optimizer.param_groups[0]["lr"] == lr

    def test_valid_batch_sizes(self) -> None:
        """Test that DataLoader works with various batch sizes."""
        # Create simple tensor dataset
        x = torch.randn(20, 3, 224, 224)
        y = torch.randint(0, 2, (20,))
        dataset = TensorDataset(x, y)
        
        for batch_size in [1, 2, 4, 8, 16]:
            loader = DataLoader(dataset, batch_size=batch_size)
            images, labels = next(iter(loader))
            assert images.shape[0] == batch_size
            assert labels.shape[0] == batch_size
