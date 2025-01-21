from typing import Optional

import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.simplenet_learner.models.backborn import Backborn
from src.simplenet_learner.models.optimizers import get_optimizer
from src.simplenet_learner.models.schedulers import get_lr_sheduler
from src.simplenet_learner.utils.noise_generator import NoiseGenerator
from src.simplenet_learner.utils.patches import extract_patches_from_tensor


class SimpleNetModule(LightningModule):
    def __init__(
        self,
        backborn: Backborn,
        backborn_layers_cfg: DictConfig,
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
        backborn_layers = OmegaConf.to_container(backborn_layers_cfg)
        self.backborn = backborn.get_feature_extraction_model(
            layers_dict=backborn_layers
        )
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

    def _embed(self, images: torch.Tensor):
        features_dict: dict[str, torch.Tensor] = self.backborn(images)
        features = [features_dict[layer] for layer in features_dict]

        patches = [extract_patches_from_tensor(f) for f in features]
        patch_shapes = []

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
