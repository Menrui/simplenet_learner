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

logger = get_logger(__name__)


def training_pipeline(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

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
        logger=lightning_loggers,
        _convert_="partial",
        enable_checkpointing=True,
    )
    logger.info(trainer.checkpoint_callbacks)

    logger.info("Logging hyperparameters:")
    log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=lightning_loggers,
    )

    if config.get("train"):
        logger.info("Start training")
        trainer.fit(model, datamodule)
    logger.info(f"callback_metrics: {list(trainer.callback_metrics.keys())}")
    logger.info(f"logged_metrics: {list(trainer.logged_metrics.keys())}")

    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    if config.get("test"):
        logger.info("Start testing")
        ckpt_path = None
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if not config.trainer.get("fast_dev_run") and config.get("train"):
        logger.info(
            f"Best model ckpt at {trainer.checkpoint_callbacks[0].best_model_path}"
        )

    return score
