import json
from pathlib import Path
from typing import Union

from hydra import compose, initialize

from simplenet_learner.simplenet_get_onnx_statistics_pipeline import (
    get_onnx_statistics_pipeline,
)
from simplenet_learner.simplenet_get_torch_statistics_pipeline import (
    get_torch_statistics_pipeline,
)


def argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Predict the model")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/predict.yaml",
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
        "--input_dir", "-i", type=str, default="input/", help="input directory path"
    )
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model")
    return parser.parse_args()


def predict(
    config_path: Union[str, Path],
    ckpt_path: Union[str, Path],
    input_dir: Union[str, Path],
    is_onnx: bool = False,
):
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    with initialize(config_path=str(config_path.parent), version_base="1.1"):
        config = compose(config_name=str(config_path.stem), return_hydra_config=True)
        if is_onnx:
            output = get_onnx_statistics_pipeline(
                config,
                ckpt_path,
                input_dir,
                resize_shape=(
                    config.datamodule.transform_cfg.crop_h,
                    config.datamodule.transform_cfg.crop_w,
                ),
            )
        else:
            output = get_torch_statistics_pipeline(
                config,
                ckpt_path,
                input_dir,
                resize_shape=(
                    config.datamodule.transform_cfg.crop_h,
                    config.datamodule.transform_cfg.crop_w,
                ),
            )

    with open(ckpt_path.parent / "statistics.json", "w") as f:
        json.dump(output, f)
    return output


if __name__ == "__main__":
    args = argparser()
    output = predict(args.config, args.ckpt_path, args.input_dir, args.onnx)
    print(output)
