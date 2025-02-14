import numpy as np
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from scipy import ndimage

from simplenet_learner.models.networks.simplenet2d import Simplenet2D
from simplenet_learner.models.optimizers import get_optimizer
from simplenet_learner.models.schedulers import get_lr_scheduler
from simplenet_learner.utils.metric import (
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from simplenet_learner.utils.noise_generator import NoiseGenerator2D
from simplenet_learner.utils.patch_utils import compute_image_score_from_patches


class Simplenet2DModule(LightningModule):
    def __init__(
        self,
        simplenet2d_cfg: DictConfig,
        projection_optimizer_cfg: DictConfig,
        projection_lr_scheduler_cfg: DictConfig,
        descriminator_optimizer_cfg: DictConfig,
        descriminator_lr_scheduler_cfg: DictConfig,
        noise_generator: NoiseGenerator2D,
        anomaly_threshold: float = 0.8,
        size_of_predict_mask: tuple[int, int] = (224, 224),
    ):
        super().__init__()

        self.model = Simplenet2D(
            backborn_arch=simplenet2d_cfg.backborn_arch,
            backborn_pretrained=simplenet2d_cfg.backborn_pretrained,
            backborn_trainable=simplenet2d_cfg.backborn_trainable,
            projection_channel=simplenet2d_cfg.projection_channel,
            projection_layer_num=simplenet2d_cfg.projection_layer_num,
            descriminator_layer_num=simplenet2d_cfg.descriminator_layer_num,
            descriminator_reduce_rate=simplenet2d_cfg.descriminator_reduce_rate,
        )

        if self.model.projection is not None:
            self.projection_optimizer = get_optimizer(
                self.model.projection, projection_optimizer_cfg
            )
            self.projection_lr_scheduler = get_lr_scheduler(
                self.projection_optimizer, projection_lr_scheduler_cfg
            )
        self.descriminator_optimizer = get_optimizer(
            self.model.descriminator, descriminator_optimizer_cfg
        )
        self.descriminator_lr_scheduler = get_lr_scheduler(
            self.descriminator_optimizer, descriminator_lr_scheduler_cfg
        )

        self.anomaly_threshold = anomaly_threshold
        self.noise_generator = noise_generator
        self.size_of_predict_mask = size_of_predict_mask

        self.training_step_outputs: list[dict[str, torch.Tensor]] = []
        self.test_step_outputs: list[dict[str, torch.Tensor]] = []

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        if self.model.projection is not None:
            projection_optimizer, descriminator_optimizer = self.configure_optimizers()
            projection_optimizer.zero_grad()
        else:
            descriminator_optimizer = self.configure_optimizers()
        descriminator_optimizer.zero_grad()

        x = batch[0]
        # Feature extraction and Projection
        true_features = self.model.forward_features(x)
        fake_features = self.noise_generator.add_noise(true_features)

        # Descriminator
        true_scores = self.model.forward_descriminator(true_features)
        fake_scores = self.model.forward_descriminator(fake_features)
        _, _, score_h, score_w = true_scores.shape

        # Loss
        true_loss = torch.clip(-true_scores + self.anomaly_threshold, min=0)
        fake_loss = torch.clip(fake_scores + self.anomaly_threshold, min=0)
        loss = true_loss.mean() + fake_loss.mean()
        loss.backward()

        # Optimizer step
        if self.model.projection is not None:
            projection_optimizer.step()
        descriminator_optimizer.step()

        # Scheduler step
        if self.projection_lr_scheduler is not None:
            self.projection_lr_scheduler.step()
        if self.descriminator_lr_scheduler is not None:
            self.descriminator_lr_scheduler.step()

        # Metric
        p_true = (true_scores >= self.anomaly_threshold).sum() / (
            len(true_scores) * score_h * score_w
        )
        p_fake = (fake_scores >= self.anomaly_threshold).sum() / (
            len(fake_scores) * score_h * score_w
        )

        # Log
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        # self.log("p_true", p_true, prog_bar=False, on_epoch=True)
        # self.log("p_fake", p_fake, prog_bar=False, on_epoch=True)
        self.training_step_outputs.append(
            {
                "loss": loss,
                "p_true": p_true,
                "p_fake": p_fake,
            }
        )

    def on_training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return avg_loss

    def configure_optimizers(self):
        if self.model.projection is not None:
            return self.projection_optimizer, self.descriminator_optimizer
        return self.descriminator_optimizer

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        true_features = self.model.forward_features(x)
        fake_features = self.noise_generator.add_noise(true_features)

        true_scores = self.model.forward_descriminator(true_features)
        fake_scores = self.model.forward_descriminator(fake_features)
        _, _, score_h, score_w = true_scores.shape

        true_loss = torch.clip(-true_scores + self.anomaly_threshold, min=0)
        fake_loss = torch.clip(fake_scores + self.anomaly_threshold, min=0)
        loss = true_loss.mean() + fake_loss.mean()

        p_true = (true_scores >= self.anomaly_threshold).sum() / (
            len(true_scores) * score_h * score_w
        )
        p_fake = (fake_scores >= self.anomaly_threshold).sum() / (
            len(fake_scores) * score_h * score_w
        )

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_p_true", p_true, prog_bar=True, on_epoch=True)
        self.log("val_p_fake", p_fake, prog_bar=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        self.trainer.save_checkpoint("last.ckpt")
        return super().on_validation_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x = batch[0]
        score_maps = -self.model.forward(x)
        batch_size, _, score_h, score_w = score_maps.shape
        score_maps = score_maps.reshape(batch_size, score_h, score_w)
        image_scores = compute_image_score_from_patches(score_maps.cpu().numpy())
        # image_scores = score_maps.amax(dim=(-1, -2)).cpu().numpy()

        target_h, target_w = self.size_of_predict_mask
        resized_smoothing_scores = []
        for i in range(batch_size):
            resized_smoothing_scores.append(
                torch.nn.functional.interpolate(
                    score_maps[i].unsqueeze(0).unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        resized_smoothing_scores = [
            ndimage.gaussian_filter(x, sigma=1) for x in resized_smoothing_scores
        ]

        return list(image_scores), list(resized_smoothing_scores)

    def evalueate(
        self,
        image_scores: list[float],
        score_maps: list[np.ndarray],
        gt_image_labels: list[np.ndarray],
        gt_masks: list[np.ndarray],
    ) -> tuple[float, float]:
        image_scores_np = np.squeeze(np.array(image_scores))
        image_min_score = image_scores_np.min(axis=-1)
        image_max_score = image_scores_np.max(axis=-1)
        normalized_image_scores = (image_scores_np - image_min_score) / (
            image_max_score - image_min_score
        )

        auroc = compute_imagewise_retrieval_metrics(
            anomaly_prediction_weights=normalized_image_scores,
            anomaly_ground_truth_labels=gt_image_labels,
        )["auroc"]

        if len(score_maps) <= 0:
            pixelwise_auroc = -1
        else:
            score_maps_np = np.array(score_maps)
            min_scores = (
                score_maps_np.reshape(len(score_maps_np), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                score_maps_np.reshape(len(score_maps_np), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            normalized_score_maps = np.zeros_like(score_maps_np)

            for i in range(len(score_maps_np)):
                normalized_score_maps[i] = (score_maps_np[i] - min_scores[i]) / (
                    max_scores[i] - min_scores[i]
                )
            # print(normalized_score_maps.shape)
            # print(normalized_score_maps.max(), normalized_score_maps.min())

            pixelwise_auroc = compute_pixelwise_retrieval_metrics(
                anomaly_segmentations=normalized_score_maps,
                ground_truth_masks=gt_masks,
            )["auroc"]

        return auroc, pixelwise_auroc

    def test_step(self, batch, batch_idx):
        gt_masks = batch[1]
        gt_labels = (gt_masks.reshape(gt_masks.shape[0], -1) > 0).any(dim=1).long()
        image_scores, score_maps = self.predict_step(batch, batch_idx, 0)

        self.test_step_outputs.append(
            {
                "image_scores": image_scores,
                "score_maps": score_maps,
                "gt_labels": gt_labels.cpu().numpy().tolist(),
                "gt_masks": gt_masks.cpu().numpy(),
            }
        )

        return image_scores, score_maps

    def on_test_epoch_end(self):
        image_scores = []
        score_maps = []
        gt_labels = []
        gt_masks = []
        for output in self.test_step_outputs:
            image_scores.extend(output["image_scores"])
            score_maps.extend(output["score_maps"])
            gt_labels.extend(output["gt_labels"])
            gt_masks.extend(output["gt_masks"])

        auroc, pixelwise_auroc = self.evalueate(
            image_scores, score_maps, gt_labels, gt_masks
        )

        self.log("auroc", auroc, prog_bar=True, on_epoch=True)
        self.log("pixelwise_auroc", pixelwise_auroc, prog_bar=True, on_epoch=True)
        self.test_step_outputs = []
        return auroc
