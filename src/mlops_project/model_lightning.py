from torch import nn
import torch
from pytorch_lightning import LightningModule


class LitModel(LightningModule):
    """CNN model for chest X-ray pneumonia classification."""

    def __init__(self, num_classes: int = 2, learning_rate: float = 1e-4) -> None:
        """Initialize the CNN model.

        Args:
            num_classes: Number of output classes (default: 2 for NORMAL/PNEUMONIA)
            learning_rate: Learning rate for the optimizer (default: 1e-4)
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        # After 4 pooling operations: 224 -> 112 -> 56 -> 28 -> 14
        # So the feature map size is 14x14
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Activation function
        self.relu = nn.ReLU()
    
    def load_from_checkpoint(cls, checkpoint_path: str) -> "LitModel":
        """Load model from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.
        Returns:
            Loaded Model instance.
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model = cls(num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolutional block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolutional block 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolutional block 4
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten the feature maps
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers for PyTorch Lightning."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    model = LitModel()
    # Test with a batch of images (batch_size=4, channels=3, height=224, width=224)
    x = torch.rand(4, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
