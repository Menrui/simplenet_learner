from pathlib import Path
from typing import Optional

import hydra
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from simplenet_learner.datamodules.components.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def predict_pipeline(
    config: DictConfig,
    ckpt_path: Path,
    input_data: Image,
    resize_shape: tuple[int, int],
) -> Optional[float]:
    """Contains the prediction pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        ckpt_path (Path): Path to the checkpoint file.
        input_data (torch.Tensor): Input data for prediction.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    input_transform = transforms.Compose(
        [
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    model: LightningModule = hydra.utils.instantiate(config.model)
    model.load_from_checkpoint(ckpt_path)
    model.eval()

    with torch.no_grad():
        input_data_tensor = input_transform(input_data)
        input_data_tensor = input_data_tensor.unsqueeze(0)
        output = model(input_data)

    return output
