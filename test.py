from pathlib import Path
from typing import Union
from hydra import compose, initialize

from simplenet_learner.testing_pipline import testing_pipline

def argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--config', "-c", type=str, default='config/test.yaml', help='config file')
    parser.add_argument('--ckpt_path', "-p", type=str, default='model.ckpt', help='model checkpoint')
    return parser.parse_args()

def test(config_path: Union[str, Path], ckpt_path: Union[str, Path]):
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)

    with initialize(config_path=str(config_path.parent), version_base="1.1"):
        config = compose(config_name=str(config_path.stem), return_hydra_config=True)
        testing_pipline(config, ckpt_path)

if __name__ == "__main__":
    args = argparser()
    test(args.config, args.ckpt_path)