from pathlib import Path
from typing import Union
from hydra import compose, initialize
from PIL import Image

from simplenet_learner.simplenet_predict_pipeline import predict_pipeline

def argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Predict the model')
    parser.add_argument('--config', "-c", type=str, default='config/predict.yaml', help='config file path')
    parser.add_argument('--ckpt_path', "-p", type=str, default='model.ckpt', help='model checkpoint path')
    parser.add_argument('--input', "-i", type=str, default='input.png', help='input image path')
    return parser.parse_args()

def predict(config_path: Union[str, Path], ckpt_path: Union[str, Path], input_path: Union[str, Path]):
    if isinstance(config_path, str):
        config_path = Path(config_path)
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    if isinstance(input_path, str):
        input_path = Path(input_path)

    with initialize(config_path=str(config_path.parent), version_base="1.1"):
        config = compose(config_name=str(config_path.stem), return_hydra_config=True)
        input_data = Image.open(input_path)
        output = predict_pipeline(config, ckpt_path, input_data, resize_shape=(224, 224))
    
    return output

if __name__ == "__main__":
    args = argparser()
    output = predict(args.config, args.ckpt_path, args.input)
    print(output[0])
    print(output[1])
    print(output[2])