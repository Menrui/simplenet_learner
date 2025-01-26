from pathlib import Path
from typing import Optional, Union

import numpy as np
import hydra
from simplenet_learner.models.simplenet import SimpleNetModule
import torch
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
    input_data: Union[Image, np.ndarray],
    resize_shape: tuple[int, int],
) -> Optional[float]:
    """Contains the prediction pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        ckpt_path (Path): Path to the checkpoint file.
        input_data (Union[PIL.Image, np.ndarray]): Input data for prediction.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    if isinstance(input_data, np.ndarray):
        input_data = Image.fromarray(input_data)
    input_transform = transforms.Compose(
        [
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    ckpt = torch.load(ckpt_path)
    full_state_dict = ckpt["state_dict"]
    backborn_dict = {}
    for k, v in full_state_dict.items():
        # k は "backborn.xxx" や "descriminator.xxx" のようにLightningModuleから見た階層名が含まれる
        if k.startswith("backborn."):
            # `load_state_dict` 用にキーから "backborn." を取り除いたほうが良い場合が多い
            new_key = k.replace("backborn.", "")
            backborn_dict[new_key] = v
    projection_dict = {k.replace("projection.", ""): v for k, v in full_state_dict.items() if k.startswith("projection.")}
    descriminator_dict = {k.replace("descriminator.", ""): v for k, v in full_state_dict.items() if k.startswith("descriminator.")}

    model: SimpleNetModule = hydra.utils.instantiate(config.model)
    model.backborn.load_state_dict(backborn_dict, strict=False)
    model.projection.load_state_dict(projection_dict, strict=False)
    model.descriminator.load_state_dict(descriminator_dict, strict=False)
    model.eval()

    with torch.no_grad():
        input_data_tensor = input_transform(input_data)
        input_data_tensor = input_data_tensor.unsqueeze(0)
        output = model(input_data_tensor)

    return output
