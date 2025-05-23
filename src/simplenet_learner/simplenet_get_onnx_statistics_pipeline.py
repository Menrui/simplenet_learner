from pathlib import Path
from typing import Optional, Union

import numpy as np
import onnxruntime as ort
import torch
from omegaconf import DictConfig
from torchvision import transforms
from tqdm import tqdm

from simplenet_learner.datamodules.components.directory_image import (
    DirectoryImageDataset,
)
from simplenet_learner.datamodules.components.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_onnx_statistics_pipeline(
    config: DictConfig,
    onnx_model_path: Path,
    input_data_dir: Union[str, Path],
    resize_shape: tuple[int, int],
) -> Optional[dict[str, float]]:
    """Contains the prediction pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        ckpt_path (Path): Path to the onnx file.
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

    model = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    dataset = DirectoryImageDataset(str(input_data_dir), transform=input_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    segmentations: list[np.ndarray] = []
    with torch.no_grad():
        for input_data in tqdm(dataloader):
            input_data = input_data.numpy()
            mask = np.array(
                model.run(
                    [output_name],
                    {input_name: input_data},
                )[0]
            )
            segmentations.extend(mask)

    # calculate statistics
    segmentations_np = np.array(segmentations)
    print(segmentations_np.shape)
    output = {
        "mean": float(np.mean(segmentations_np)),
        "std": float(np.std(segmentations_np)),
        "max": float(np.max(segmentations_np)),
        "min": float(np.min(segmentations_np)),
    }

    return output
