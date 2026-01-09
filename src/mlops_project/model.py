from torch import nn
import torch


class Model(nn.Module):
    """CNN model for chest X-ray pneumonia classification."""

    def __init__(self, num_classes: int = 2) -> None:
        """Initialize the CNN model.

        Args:
            num_classes: Number of output classes (default: 2 for NORMAL/PNEUMONIA)
        """
        super().__init__()
        self.num_classes = num_classes

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


if __name__ == "__main__":
    model = Model()
    # Test with a batch of images (batch_size=4, channels=3, height=224, width=224)
    x = torch.rand(4, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
