from pathlib import Path
import time
from typing import Optional, Union

import numpy as np
import onnxruntime as ort
import torch
from omegaconf import DictConfig
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from simplenet_learner.datamodules.components.directory_image import (
    DirectoryImageDataset,
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def preprocess_image(image: np.ndarray, resize_shape: tuple = None) -> np.ndarray:
    """
    torchvisionのtransformsと同等の前処理をNumPyで実装

    Args:
        image (np.ndarray): (H, W, C)形式のint8画像
        resize_shape (tuple, optional): リサイズ後のサイズ (height, width)

    Returns:
        np.ndarray: 正規化済み画像 (1, C, H, W)
    """

    # print("image", image.shape)
    # print("resize_shape", resize_shape)
    if resize_shape is not None:
        # リサイズ
        image = Image.fromarray(image.squeeze()).resize(resize_shape[::-1], Image.BILINEAR)
        image = np.expand_dims(np.array(image), axis=-1)  # (H, W, C) -> (H, W, C, 1)

    # float32に変換して0-1の範囲にスケーリング
    image = image.astype(np.float32) / 255.0

    # (W, H, C) から (C, H, W) に転置
    if image.ndim == 2:
        # グレースケール画像の場合、チャネルを追加
        image = np.expand_dims(image, axis=-1)
    image = image.transpose(2, 0, 1)

    # グレースケール画像の場合、チャネルを複製
    if image.shape[0] == 1:
        image = np.concatenate([image] * 3, axis=0)

    # バッチ次元を追加 (1, C, H, W)
    image = np.expand_dims(image, axis=0)

    # ImageNetの平均と標準偏差で正規化
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image


def measure_onnx_process_time_pipline(
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

    # input_transform = transforms.Compose(
    #     [
    #         transforms.Resize(resize_shape),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    #     ]
    # )

    onnx_option = ort.SessionOptions()
    # onnx_option.log_severity_level = 0
    gpu_model = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CUDAExecutionProvider"],
        sess_options=onnx_option,
    )
    cpu_model = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"],
        sess_options=onnx_option,
    )
    # Get input and output names
    input_name = gpu_model.get_inputs()[0].name
    output_name = gpu_model.get_outputs()[0].name

    # dataset = DirectoryImageDataset(str(input_data_dir), transform=input_transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # load first image using pillow

    # Find all image files in input_data_dir (common image extensions)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    input_data_dir = Path(input_data_dir)
    image_paths = sorted([p for p in input_data_dir.rglob('*') if p.suffix.lower() in image_extensions])
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {input_data_dir}")
    # Load the first image using PIL.Image
    img = np.array(Image.open(image_paths[0]))
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    print(f"Loaded image: {image_paths[0]}")
    # data = np.random.rand(1, 3, resize_shape[0], resize_shape[1]).astype(np.float32)

    # Measure the time taken for inference
    preprocessing_times = []
    gpu_times = []
    cpu_times = []
    for _ in tqdm(range(100)):
        t = time.perf_counter()
        data = preprocess_image(img, resize_shape)
        preprocessing_times.append(time.perf_counter() - t)

        input_data = {input_name: data}
        t = time.perf_counter()
        _ = gpu_model.run([output_name], input_data)
        gpu_times.append(time.perf_counter() - t)
        time.sleep(0.1)  # Simulate some delay for CPU processing

        t = time.perf_counter()
        _ = cpu_model.run([output_name], input_data)
        cpu_times.append(time.perf_counter() - t)

    preprocessing_avg_time = np.mean(preprocessing_times)
    gpu_avg_time = np.mean(gpu_times)
    cpu_avg_time = np.mean(cpu_times)
    print(f"Preprocessing average time: {preprocessing_avg_time:.6f} seconds.")
    print(f"GPU average time: {gpu_avg_time:.6f} seconds. providers: {gpu_model.get_providers()}")
    print(f"CPU average time: {cpu_avg_time:.6f} seconds. providers: {cpu_model.get_providers()}")
    output = {
        "gpu_avg_time": gpu_avg_time,
        "cpu_avg_time": cpu_avg_time,
    }

    return output
