from typing import Any, List, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer


@hydra.main(config_path="../configs", config_name="train", version_base="1.2")
def train(cfg: DictConfig) -> Optional[float]:
    """Execute the main training routine.

    Args:
    ----
        cfg: Configuration composed by Hydra.

    Returns:
    -------
        Optional float: Metric score for hyperparameter optimization.
    """
    # Set seed for random number generators
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Initialize data module
    data_module: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Initialize model
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Initialize callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if cb_conf is not None:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Initialize logger
    logger = hydra.utils.instantiate(cfg.logger)  # type: ignore

    # Initialize trainer
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Train the model
    trainer.fit(model=model, datamodule=data_module)

    # Test the model if configured
    if cfg.get("test"):
        trainer.test(model=model, datamodule=data_module)

    # Get metric score for hyperparameter optimization
    metric: Any = trainer.callback_metrics.get("val/acc_best")

    # Handle different metric types
    if metric is None:
        return None

    # Convert tensor to float
    if isinstance(metric, torch.Tensor):
        return float(metric.item())

    # Convert numeric types to float
    if isinstance(metric, (int, float)):
        return float(metric)

    return None


if __name__ == "__main__":
    train()
