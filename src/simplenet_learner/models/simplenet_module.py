from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import nn

from simplenet_learner.models.networks.backborn import Backborn
from simplenet_learner.models.optimizers import get_optimizer
from simplenet_learner.models.schedulers import get_lr_scheduler
from simplenet_learner.utils.embed_postprocessor import RescaleSegmentator
from simplenet_learner.utils.embed_preprocessor import (
    EmbeddingAggregator,
    EmbeddingPreprocessor,
)
from simplenet_learner.utils.metric import (
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from simplenet_learner.utils.noise_generator import NoiseGenerator
from simplenet_learner.utils.patch_utils import (
    compute_image_score_from_patches,
    extract_patches_from_tensor,
    rearrange_patches_to_batch,
)


class OriginalSimplenetModule(LightningModule):
    def __init__(
        self,
        backborn: Backborn,
        backborn_layers_cfg: DictConfig,
        projection: Optional[nn.Module],
        descriminator: nn.Module,
        embed_preprocessor_cfg: DictConfig,
        embed_aggregator_cfg: DictConfig,
        projection_optimizer_cfg: DictConfig,
        projection_lr_scheduler_cfg: DictConfig,
        descriminator_optimizer_cfg: DictConfig,
        descriminator_lr_scheduler_cfg: DictConfig,
        noise_generator: NoiseGenerator,
        anomaly_threshold: float = 0.8,
        size_of_predict_mask: tuple[int, int] = (224, 224),
    ):
        super().__init__()

        # networks
        backborn_layers: dict[str, str] = OmegaConf.to_container(backborn_layers_cfg)
        self.backborn = backborn.get_feature_extraction_model(
            layers_dict=backborn_layers
        )
        self.backborn.requires_grad_(False)
        self.projection: Optional[nn.Module] = projection
        self.descriminator = descriminator

        # loss_fn
        self.anomaly_threshold = anomaly_threshold

        # optimizers
        if self.projection is not None:
            self.projection_optimizer = get_optimizer(
                self.projection, projection_optimizer_cfg
            )
            self.projection_lr_scheduler = get_lr_scheduler(
                self.projection_optimizer, projection_lr_scheduler_cfg
            )
        self.descriminator_optimizer = get_optimizer(
            self.descriminator, descriminator_optimizer_cfg
        )
        self.descriminator_lr_scheduler = get_lr_scheduler(
            self.descriminator_optimizer, descriminator_lr_scheduler_cfg
        )

        # preprocessors
        self.embed_preprocessor = EmbeddingPreprocessor(
            input_dims=self.get_backborn_feature_dimention_list(
                input_shape=(3, *size_of_predict_mask)
            ),
            output_dim=embed_preprocessor_cfg.output_dim,
        )
        self.embed_aggregator = EmbeddingAggregator(
            target_dim=embed_aggregator_cfg.output_dim
        )

        # utils
        self.noise_generator = noise_generator
        self.rescale_segmentator = RescaleSegmentator(
            device=torch.device("cpu"), target_size=list(size_of_predict_mask)
        )

        self.automatic_optimization = False
        self.training_step_outputs: list[dict[str, torch.Tensor]] = []
        self.validation_step_outputs: list[float] = []
        self.test_step_outputs: list[dict[str, torch.Tensor]] = []

    def get_backborn_feature_dimention_list(
        self, input_shape: tuple[int, int, int]
    ) -> list[int]:
        x = torch.ones([1] + list(input_shape)).to(self.device)
        features_dict: dict[str, torch.Tensor] = self.backborn(x)
        features = [features_dict[out_tag] for out_tag in features_dict]
        return [f.shape[1] for f in features]

    def forward(self, x):
        image_scores, segmentation_masks, interpolated_features = self.predict_step(
            (x, None), 0
        )
        return image_scores, segmentation_masks, interpolated_features

    def _embed(self, images: torch.Tensor):
        features_dict: dict[str, torch.Tensor] = self.backborn(images)
        # with torch.no_grad():
        features = [features_dict[layer] for layer in features_dict]

        patches: list[tuple[torch.Tensor, list[int]]] = [
            extract_patches_from_tensor(f, return_spatial_info=True) for f in features
        ]
        features = [
            x[0] for x in patches
        ]  # (B, N_patches, C, patch_size, patch_size)  N_patches = num_patches_height * num_patches_width
        patch_shapes = [
            x[1] for x in patches
        ]  # [(num_patches_height, num_patches_width), ...]
        reference_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0],  # batch_size
                patch_dims[0],  # N_patches_height
                patch_dims[1],  # N_patches_width
                *_features.shape[2:],  # C, patch_size, patch_size
            )
            # heightとwidthを最後の次元に移動
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            permute_base_shape = _features.shape
            _features = _features.reshape(
                -1,
                *_features.shape[-2:],  # N_patches_height, N_patches_width
            )
            _features = F.interpolate(
                _features.unsqueeze(1),  # (B*C, 1, N_patches_height, N_patches_width)
                size=reference_num_patches,  # (N_patches_height, N_patches_width)
                mode="bilinear",
                align_corners=False,
            )
            # サイズを補完後に次元の順序を戻す
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *permute_base_shape[:-2],
                reference_num_patches[0],
                reference_num_patches[1],
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(
                len(_features),  # batch_size
                -1,  # N_patches
                *_features.shape[-3:],  # C, patch_size, patch_size
            )
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        features = self.embed_preprocessor(features)
        features = self.embed_aggregator(features)

        return features, patch_shapes

    def training_step(self, batch, batch_idx):
        # get optimizer
        if self.projection is not None:
            projection_optimizer, descriminator_optimizer = self.configure_optimizers()
            projection_optimizer.zero_grad()
        else:
            descriminator_optimizer = self.configure_optimizers()

        descriminator_optimizer.zero_grad()

        # forward
        x = batch[0]
        embedding, _ = self._embed(x)
        true_features = self.projection(embedding)
        noise = self.noise_generator(true_features)
        fake_features = true_features + noise

        # descriminator
        true_scores = self.descriminator(true_features)
        fake_scores = self.descriminator(fake_features)

        # loss
        p_true = (true_scores >= self.anomaly_threshold).sum() / len(true_scores)
        p_fake = (fake_scores >= self.anomaly_threshold).sum() / len(fake_scores)
        true_loss = torch.clip(-true_scores + self.anomaly_threshold, min=0)
        fake_loss = torch.clip(fake_scores - self.anomaly_threshold, min=0)

        loss = true_loss.mean() + fake_loss.mean()
        loss.backward()

        if self.projection is not None:
            projection_optimizer.step()
        descriminator_optimizer.step()

        # scheduler
        if self.projection_lr_scheduler is not None:
            self.projection_lr_scheduler.step()
        if self.descriminator_lr_scheduler is not None:
            self.descriminator_lr_scheduler.step()

        self.training_step_outputs.append(
            {
                "loss": loss,
                "p_true": p_true,
                "p_fake": p_fake,
            }
        )

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("p_true", p_true, prog_bar=False, on_step=False, on_epoch=True)
        self.log("p_fake", p_fake, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        if self.projection is not None:
            return self.projection_optimizer, self.descriminator_optimizer
        return self.descriminator_optimizer

    def on_training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return avg_loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        embedding, _ = self._embed(x)
        true_features = self.projection(embedding)
        # fake_features = self.noise_generator.add_noise(true_features)
        noise = self.noise_generator(true_features)
        fake_features = true_features + noise

        true_scores = self.descriminator(true_features)
        fake_scores = self.descriminator(fake_features)

        p_true = (true_scores.detach() >= self.anomaly_threshold).sum() / len(
            true_scores
        )
        p_fake = (fake_scores.detach() >= self.anomaly_threshold).sum() / len(
            fake_scores
        )
        true_loss = torch.clip(-p_true + self.anomaly_threshold, min=0)
        fake_loss = torch.clip(p_fake - self.anomaly_threshold, min=0)

        loss = true_loss.mean() + fake_loss.mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        self.trainer.save_checkpoint("last.ckpt")

    def test_step(self, batch, batch_idx):
        x = batch[0]
        gt_masks: torch.Tensor = batch[1]
        gt_labels = (gt_masks.reshape(gt_masks.shape[0], -1) > 0).any(dim=1).long()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        image_scores, segmentation_masks, interpolated_features = self.predict_step(
            batch, batch_idx
        )

        self.test_step_outputs.append(
            {
                "gt_masks": gt_masks.cpu().numpy(),
                "gt_labels": gt_labels.cpu().numpy().tolist(),
                "image_scores": image_scores,
                "segmentation_masks": segmentation_masks,
                "interpolated_features": interpolated_features,
            }
        )

        return image_scores, segmentation_masks, interpolated_features

    def on_test_epoch_end(self):
        image_scores = []
        segmentation_masks = []
        interpolated_features = []
        gt_masks = []
        gt_labels = []
        for output in self.test_step_outputs:
            image_scores.extend(output["image_scores"])
            segmentation_masks.extend(output["segmentation_masks"])
            interpolated_features.extend(output["interpolated_features"])
            gt_masks.extend(output["gt_masks"])
            gt_labels.extend(output["gt_labels"])

        # calculate metrics
        auroc, pixelwise_auroc = self.evaluate(
            image_scores=image_scores,
            segmentation_masks=segmentation_masks,
            gt_image_labels=gt_labels,
            gt_segmentation_masks=gt_masks,
        )
        self.log("auroc", auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "pixelwise_auroc",
            pixelwise_auroc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.test_step_outputs.clear()
        return image_scores

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[0]
        batch_size = len(x)
        embedding, patch_shapes = self._embed(x)
        features = self.projection(embedding)
        patch_scores = image_scores = -self.descriminator(features)

        image_scores = rearrange_patches_to_batch(
            patches=patch_scores, batchsize=batch_size
        )
        image_scores = image_scores.cpu().numpy().reshape(*image_scores.shape[:2], -1)
        image_scores = compute_image_score_from_patches(patches=image_scores, top_k=1)

        patch_scores = rearrange_patches_to_batch(
            patches=patch_scores, batchsize=batch_size
        )
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batch_size, scales[0], scales[1])
        features = features.reshape(batch_size, scales[0], scales[1], -1)

        segmentation_masks, interpolated_features = (
            self.rescale_segmentator.convert_to_segmentation(
                patch_scores=patch_scores, features=features
            )
        )

        return list(image_scores), list(segmentation_masks), list(interpolated_features)

    def evaluate(
        self,
        image_scores: list[float],
        segmentation_masks: list[np.ndarray],
        gt_image_labels: list[np.ndarray],
        gt_segmentation_masks: list[np.ndarray],
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

        if len(segmentation_masks) <= 0:
            pixelwise_auroc = -1
        else:
            segmentation_masks_np = np.array(segmentation_masks)
            min_scores = (
                segmentation_masks_np.reshape(len(segmentation_masks_np), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentation_masks_np.reshape(len(segmentation_masks_np), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            normalized_segmentation_masks = np.zeros_like(segmentation_masks_np)

            for i in range(len(segmentation_masks_np)):
                normalized_segmentation_masks[i] = (
                    segmentation_masks_np[i] - min_scores[i]
                ) / (max_scores[i] - min_scores[i])
            print(normalized_segmentation_masks.shape)
            print(
                normalized_segmentation_masks.max(), normalized_segmentation_masks.min()
            )

            pixelwise_auroc = compute_pixelwise_retrieval_metrics(
                anomaly_segmentations=normalized_segmentation_masks,
                ground_truth_masks=gt_segmentation_masks,
            )["auroc"]

        return auroc, pixelwise_auroc
