import os
from pathlib import Path
from typing import Union

import dotenv
from hydra import compose, initialize

from simplenet_learner.simplenet2d_torch2onnx import torch2onnx_pipeline

dotenv.load_dotenv()


def argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Convert the model to ONNX")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/torch2onnx.yaml",
        help="config file path",
    )
    parser.add_argument(
        "--ckpt_path",
        "-p",
        type=str,
        default="model.ckpt",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--channel_last",
        "-l",
        action="store_true",
        help="Whether to use channel last format",
    )
    return parser.parse_args()


def torch2onnx(
    config_path: Union[str, Path],
    ckpt_path: Union[str, Path],
    is_channel_last: bool = False,
):
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    model_key_str = os.environ["MODEL_KEY"]
    model_key = model_key_str.encode("utf-8")

    with initialize(config_path=str(config_path.parent), version_base="1.1"):
        config = compose(config_name=str(config_path.stem), return_hydra_config=True)
        if model_key_str:
            torch2onnx_pipeline(config, str(ckpt_path), model_key, is_channel_last)
        else:
            torch2onnx_pipeline(config, str(ckpt_path), None, is_channel_last)


if __name__ == "__main__":
    args = argparser()
    torch2onnx(args.config, args.ckpt_path, args.channel_last)
