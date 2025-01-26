from pathlib import Path
from typing import Optional

import hydra
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from simplenet_learner.utils import get_logger, log_hyperparameters
from simplenet_learner.utils.logger import get_stream_logger


logger = get_stream_logger(__name__)

def testing_pipline(config: DictConfig, ckpt_path: str) -> Optional[float]:
    """
    Run testing pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    if config.get("seed") is not None:
        seed_everything(config.seed)

    logger.info(f"Initializing datamodule: {config.datamodule._target_}")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    logger.info(f"Initializing model: {config.model._target_}")
    model: LightningModule = hydra.utils.instantiate(config.model)

    logger.info("Initializing callbacks:")
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_cfg in config.callbacks.items():
            if "_target_" in cb_cfg:
                logger.info(f"  - Initializing callback: {cb_cfg._target_}")
                callbacks.append(hydra.utils.instantiate(cb_cfg))

    logger.info("Initializing lightning loggers:")
    lightning_loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                logger.info(f"  - Instantiating logger <{lg_conf._target_}>")
                lightning_loggers.append(hydra.utils.instantiate(lg_conf))

    logger.info(f"Initializing trainer: <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        # logger=lightning_loggers,
        _convert_="partial",
        enable_checkpointing=True,
        default_root_dir=Path(ckpt_path).parent,
    )
    logger.info(trainer.checkpoint_callbacks)

    if config.get("test"):
        logger.info("Start testing")
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
