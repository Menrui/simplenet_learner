from typing import Optional

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch import nn

from src.simplenet_learner.models.optimizers import get_optimizer
from src.simplenet_learner.models.schedulers import get_lr_sheduler
from src.simplenet_learner.utils.noise_generator import NoiseGenerator


class SimpleNetModule(LightningModule):
    def __init__(
        self,
        backborn: nn.Module,
        projection: Optional[nn.Module],
        descriminator: nn.Module,
        criterion,
        projection_optimizer_cfg: DictConfig,
        projection_lr_scheduler_cfg: DictConfig,
        descriminator_optimizer_cfg: DictConfig,
        descriminator_lr_scheduler_cfg: DictConfig,
        noise_generator: NoiseGenerator,
    ):
        super().__init__()

        # networks
        self.backborn = backborn
        self.projection = projection
        self.descriminator = descriminator

        # loss_fn
        self.criterion = criterion

        # optimizers
        self.projection_optimizer = get_optimizer(
            self.projection, projection_optimizer_cfg
        )
        self.projection_lr_scheduler = get_lr_sheduler(
            self.projection_optimizer, projection_lr_scheduler_cfg
        )
        self.descriminator_optimizer = get_optimizer(
            self.descriminator, descriminator_optimizer_cfg
        )
        self.descriminator_lr_scheduler = get_lr_sheduler(
            self.descriminator_optimizer, descriminator_lr_scheduler_cfg
        )

        # utils
        self.noise_generator = noise_generator

        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        return self.projection_optimizer, self.descriminator_optimizer

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", avg_loss)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        return avg_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("test_loss", avg_loss)
        return avg_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        return self.model(x)

    def predict_epoch_end(self, outputs):
        return torch.cat(outputs)
