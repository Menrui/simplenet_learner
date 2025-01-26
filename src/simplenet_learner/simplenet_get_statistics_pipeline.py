from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm
import numpy as np
import hydra
from simplenet_learner.models.simplenet import SimpleNetModule
from simplenet_learner.datamodules.components.directory_image import DirectoryImageDataset
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from simplenet_learner.datamodules.components.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_statistics_pipeline(
    config: DictConfig,
    ckpt_path: Path,
    input_data_dir: Union[str, Path],
    resize_shape: tuple[int, int],
) -> Optional[float]:
    """Contains the prediction pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        ckpt_path (Path): Path to the checkpoint file.
        input_data_dir (Union[str, Path]): Input data directory for prediction.

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

    dataset = DirectoryImageDataset(str(input_data_dir), transform=input_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    segmentations = []
    with torch.no_grad():
        for input_data in tqdm(dataloader):
            _, mask, _ = model(input_data)
            segmentations.extend(mask)
    
    # calculate statistics
    segmentations_np = np.array(segmentations)
    output = {
        "mean": np.mean(segmentations_np),
        "std": np.std(segmentations_np),
        "max": np.max(segmentations_np),
        "min": np.min(segmentations_np)
    }


    return output