from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from simplenet_learner.models.simplenet2d_module import Simplenet2DModule
from simplenet_learner.utils.logger import get_stream_logger

logger = get_stream_logger(__name__)


def torch2onnx_pipeline(config: DictConfig, ckpt_path: str) -> None:
    """
    Run testing pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        ckpt_path (str): Path to the checkpoint file.
    """
    logger.info(f"Initializing model: {config.model._target_}")
    ltmodule: Simplenet2DModule = hydra.utils.instantiate(config.model)
    model = ltmodule.model

    logger.info("Loading checkpoint:")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    full_state_dict = checkpoint["state_dict"]
    model_state_dict = {}
    for k, v in full_state_dict.items():
        if k.startswith("model."):
            new_key = k.replace("model.", "")
            model_state_dict[new_key] = v
    model.load_state_dict(model_state_dict)

    logger.info("Converting model to ONNX:")
    model.eval()
    dummy_input = torch.randn(
        1,
        3,
        config.datamodule.transform_cfg.resize_h,
        config.datamodule.transform_cfg.resize_w,
    )
    onnx_path = Path(ckpt_path).with_suffix(".onnx")
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
    logger.info(f"Model saved to: {onnx_path}")
