from pathlib import Path
from typing import Union

from hydra import compose, initialize

from simplenet_learner.simplenet2d_torch2onnx import torch2onnx_pipeline


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
    return parser.parse_args()


def torch2onnx(config_path: Union[str, Path], ckpt_path: Union[str, Path]):
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)

    with initialize(config_path=str(config_path.parent), version_base="1.1"):
        config = compose(config_name=str(config_path.stem), return_hydra_config=True)
        torch2onnx_pipeline(config, str(ckpt_path))


if __name__ == "__main__":
    args = argparser()
    torch2onnx(args.config, args.ckpt_path)
