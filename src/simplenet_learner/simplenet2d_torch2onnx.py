import io
from pathlib import Path
from typing import Optional

import hydra
import onnx
import torch
from omegaconf import DictConfig

from simplenet_learner.models.networks.onnx_bhwc_wrapper import OnnxBhwcWrapper
from simplenet_learner.models.simplenet2d_module import Simplenet2DModule
from simplenet_learner.utils.encrypt.aes_cipher import encrypt_file
from simplenet_learner.utils.logger import get_stream_logger

logger = get_stream_logger(__name__)


def torch2onnx_pipeline(
    config: DictConfig,
    ckpt_path: str,
    model_key: Optional[bytes],
    channel_last: bool = False,
) -> None:
    """
    Run testing pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        ckpt_path (str): Path to the checkpoint file.
    """
    logger.info(f"Initializing model: {config.model._target_}")
    ltmodule: Simplenet2DModule = hydra.utils.instantiate(config.model)
    model: torch.nn.Module = ltmodule.model

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
    if channel_last:
        model = OnnxBhwcWrapper(model)
        dummy_input = torch.randn(
            1,
            config.datamodule.transform_cfg.crop_h,
            config.datamodule.transform_cfg.crop_w,
            3,
        )
    else:
        dummy_input = torch.randn(
            1,
            3,
            config.datamodule.transform_cfg.crop_h,
            config.datamodule.transform_cfg.crop_w,
        )
    model.eval()
    onnx_path = Path(ckpt_path).with_suffix(".onnx")
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
    logger.info(f"Model saved to: {onnx_path}")

    if model_key is None:
        logger.info("Model key is not provided. Skip encryption.")
        return

    onnx_model = onnx.load(onnx_path)
    buffer = io.BytesIO()
    onnx.save(onnx_model, buffer)
    onnx_model_buffer = buffer.getvalue()

    encrypted_onnx_model = encrypt_file(onnx_model_buffer, model_key)
    with open(onnx_path.with_suffix(".onnx.dat"), "wb") as f:
        f.write(encrypted_onnx_model)
    logger.info(f"Encrypted model saved to: {onnx_path.with_suffix('.onnx.dat')}")
