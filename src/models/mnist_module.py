from typing import Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class MNISTLitModule(LightningModule):
    """Implement a LightningModule for MNIST classification.

    Handles training, validation, and testing for MNIST digit classification.
    Implements basic training logic and metric tracking.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Initialize the MNIST module.

        Args:
        ----
            net: Neural network model for digit classification
            optimizer: Optimizer for training the network
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # For tracking best validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the network.

        Args:
        ----
            x: Input tensor of shape (B, 1, 28, 28)

        Returns:
        -------
            Logits tensor of shape (B, 10)
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Reset best validation accuracy at the start of training."""
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
        ----
            batch: Tuple of (images, labels)

        Returns:
        -------
            Tuple of (loss, predictions, targets)
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute training step for a single batch.

        Args:
        ----
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch

        Returns:
        -------
            Computed loss value
        """
        loss, preds, targets = self.model_step(batch)
        # Update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Execute validation step for a single batch.

        Args:
        ----
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch
        """
        loss, preds, targets = self.model_step(batch)
        # Update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log best validation accuracy at the end of the validation epoch."""
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Execute test step for a single batch.

        Args:
        ----
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch
        """
        loss, preds, targets = self.model_step(batch)
        # Update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, prog_bar=True)
        self.log("test/acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Configure optimizers for training.

        Returns
        -------
            Dictionary containing the configured optimizer
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}
