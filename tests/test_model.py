import pytest
import torch
import torch.nn as nn

from src.mlops_project.model import Model


class TestModelInitialization:
    """Test model initialization and configuration."""

    def test_model_initializes_with_default_num_classes(self) -> None:
        """Test model initialization with default number of classes."""
        model = Model()
        assert isinstance(model, nn.Module)
        assert model.num_classes == 2

    def test_model_initializes_with_custom_num_classes(self) -> None:
        """Test model initialization with custom number of classes."""
        model = Model(num_classes=5)
        assert model.num_classes == 5

    def test_model_has_trainable_parameters(self) -> None:
        """Test that model has trainable parameters after initialization."""
        model = Model()
        params = list(model.parameters())
        assert len(params) > 0, "Model has no parameters"
        assert all(p.requires_grad for p in params), "Not all parameters are trainable"


class TestModelForward:
    """Test model forward pass and output shape contracts."""

    def test_forward_single_image(self) -> None:
        """Test forward pass with a single image."""
        model = Model()
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 2)

    def test_forward_batch_of_images(self) -> None:
        """Test forward pass with a batch of images."""
        model = Model()
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)
        assert output.shape == (batch_size, 2)

    def test_forward_large_batch(self) -> None:
        """Test forward pass with a larger batch."""
        model = Model()
        batch_size = 32
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)
        assert output.shape == (batch_size, 2)

    def test_forward_custom_num_classes(self) -> None:
        """Test forward pass with custom number of classes."""
        num_classes = 10
        model = Model(num_classes=num_classes)
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        assert output.shape == (4, num_classes)

    def test_output_dtype_is_float32(self) -> None:
        """Test that output tensor has float32 dtype."""
        model = Model()
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.dtype == torch.float32

    def test_output_is_finite(self) -> None:
        """Test that output contains finite values (no NaN or Inf)."""
        model = Model()
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_output_on_gpu_if_available(self) -> None:
        """Test that model can run on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = Model().cuda()
        x = torch.randn(4, 3, 224, 224).cuda()
        output = model(x)
        assert output.is_cuda
        assert output.shape == (4, 2)


class TestModelGradients:
    """Test gradient flow and backpropagation."""

    def test_gradients_flow_through_model(self) -> None:
        """Test that gradients flow through all parameters."""
        model = Model()
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"

    def test_gradients_are_nonzero(self) -> None:
        """Test that at least some gradients are non-zero."""
        model = Model()
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        loss = output.sum()
        loss.backward()

        has_nonzero_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_nonzero_grad = True
                break

        assert has_nonzero_grad, "All gradients are zero"


class TestModelTrainEvalModes:
    """Test model behavior in training and evaluation modes."""

    def test_model_default_mode_is_train(self) -> None:
        """Test that model is in training mode by default."""
        model = Model()
        assert model.training is True

    def test_model_can_switch_to_eval_mode(self) -> None:
        """Test that model can switch to evaluation mode."""
        model = Model()
        model.eval()
        assert model.training is False

    def test_model_can_switch_back_to_train_mode(self) -> None:
        """Test that model can switch back to training mode."""
        model = Model()
        model.eval()
        model.train()
        assert model.training is True

    def test_dropout_behavior_differs_in_train_eval(self) -> None:
        """Test that dropout behaves differently in train vs eval mode."""
        model = Model()
        x = torch.randn(100, 3, 224, 224)

        # In training mode, dropout should affect output
        model.train()
        torch.manual_seed(42)
        output_train_1 = model(x)
        torch.manual_seed(42)
        output_train_2 = model(x)

        # Outputs should be identical with same seed in training
        torch.testing.assert_close(output_train_1, output_train_2)

        # In eval mode, dropout should not affect output
        model.eval()
        output_eval_1 = model(x)
        output_eval_2 = model(x)

        # Outputs should be identical in eval mode without seed
        torch.testing.assert_close(output_eval_1, output_eval_2)


class TestModelDeterminism:
    """Test model determinism and reproducibility."""

    def test_same_input_produces_same_output_in_eval(self) -> None:
        """Test that same input produces same output in eval mode."""
        model = Model()
        model.eval()
        x = torch.randn(4, 3, 224, 224)

        output_1 = model(x)
        output_2 = model(x)

        torch.testing.assert_close(output_1, output_2)

    def test_different_seeds_produce_different_initializations(self) -> None:
        """Test that different random seeds produce different model initializations."""
        torch.manual_seed(42)
        model_1 = Model()

        torch.manual_seed(123)
        model_2 = Model()

        # At least some parameters should be different
        params_differ = False
        for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
            if not torch.equal(p1, p2):
                params_differ = True
                break

        assert params_differ, "Models initialized with different seeds are identical"


class TestModelParameterCount:
    """Test model parameter counts."""

    def test_model_has_trainable_parameters(self) -> None:
        """Test that model has trainable parameters."""
        model = Model()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0

    def test_all_parameters_require_grad_by_default(self) -> None:
        """Test that all parameters require gradients by default."""
        model = Model()
        for param in model.parameters():
            assert param.requires_grad is True

    def test_parameter_count_is_consistent(self) -> None:
        """Test that parameter count is consistent across multiple instantiations."""
        model_1 = Model()
        model_2 = Model()

        count_1 = sum(p.numel() for p in model_1.parameters())
        count_2 = sum(p.numel() for p in model_2.parameters())

        assert count_1 == count_2


class TestModelInputValidation:
    """Test model behavior with invalid inputs."""

    def test_forward_raises_on_wrong_channels(self) -> None:
        """Test that model raises error with wrong number of input channels."""
        model = Model()
        x = torch.randn(1, 1, 224, 224)  # Wrong: 1 channel instead of 3

        with pytest.raises(RuntimeError):
            model(x)

    def test_forward_raises_on_wrong_spatial_dims(self) -> None:
        """Test that model raises error with wrong spatial dimensions."""
        model = Model()
        x = torch.randn(1, 3, 128, 128)  # Wrong: 128x128 instead of 224x224

        with pytest.raises(RuntimeError):
            model(x)

    def test_forward_accepts_different_batch_sizes(self) -> None:
        """Test that model accepts various batch sizes."""
        model = Model()

        for batch_size in [1, 2, 8, 16, 32]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = model(x)
            assert output.shape == (batch_size, 2)


class TestModelSaveLoad:
    """Test model saving and loading."""

    def test_model_state_dict_is_serializable(self, tmp_path) -> None:
        """Test that model state_dict can be saved and loaded."""
        model = Model()
        save_path = tmp_path / "model.pth"

        # Save model
        torch.save(model.state_dict(), save_path)

        # Load model
        new_model = Model()
        new_model.load_state_dict(torch.load(save_path, weights_only=True))

        # Test that loaded model works
        x = torch.randn(1, 3, 224, 224)
        output = new_model(x)
        assert output.shape == (1, 2)

    def test_loaded_model_produces_same_output(self, tmp_path) -> None:
        """Test that loaded model produces same output as original."""
        model = Model()
        model.eval()
        save_path = tmp_path / "model.pth"

        x = torch.randn(4, 3, 224, 224)
        original_output = model(x)

        # Save and load
        torch.save(model.state_dict(), save_path)
        new_model = Model()
        new_model.load_state_dict(torch.load(save_path, weights_only=True))
        new_model.eval()

        loaded_output = new_model(x)
        torch.testing.assert_close(original_output, loaded_output)
