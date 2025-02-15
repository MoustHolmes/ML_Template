import torch
from torch import nn


class SimpleDenseNet(nn.Module):
    """A simple feedforward neural network with configurable layer sizes.

    This network consists of three hidden layers with ReLU activation functions,
    designed for classification tasks.
    """

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 128,
        output_size: int = 10,
    ) -> None:
        """Initialize the network.

        Args:
        ----
            input_size: Size of the input features (default: 784 for MNIST)
            lin1_size: Size of the first hidden layer
            lin2_size: Size of the second hidden layer
            lin3_size: Size of the third hidden layer
            output_size: Size of the output layer (default: 10 for MNIST classes)
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
        ----
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
        -------
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.model(x)
