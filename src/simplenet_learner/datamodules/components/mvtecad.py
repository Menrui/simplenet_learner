import glob
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MVTecadDataset(Dataset):
    """
    MVTec AD データセット用 Dataset クラス。

    Args:
        root (str): MVTec AD データセットのルートディレクトリパス。
        category (str): 使用するカテゴリ名 (例: 'bottle', 'cable', 'capsule', etc.)。
        phase (str): 'train' or 'validation' or 'test' のいずれかを指定。
        transform (callable, optional): 画像に対する変換処理。
        mask_transform (callable, optional): マスクに対する変換処理。

    ディレクトリ構成の例 (root が '/path/to/mvtec'、category が 'bottle' の場合):
    /path/to/mvtec/
    └── bottle
        ├── ground_truth
        │    ├── broken_large
        │    │    ├── 000_mask.png
        │    │    └── ...
        │    ├── broken_small
        │    └── ...
        ├── test
        │    ├── broken_large
        │    │    ├── 000.png
        │    │    └── ...
        │    ├── broken_small
        │    ├── contamination
        │    └── good
        └── train
             └── good
                 ├── 000.png
                 └── ...
    """

    def __init__(
        self,
        root: str,
        category: str,
        phase: str = "train",
        transform=None,
        mask_transform=None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.root = root
        self.category = category
        self.phase = phase.lower()
        self.transform = transform
        self.mask_transform = mask_transform
        self.device = device

        self.category_path = os.path.join(self.root, self.category)
        self.image_paths = []
        self.mask_paths: list[Optional[str]] = []

        # phase ごとにデータを集める
        if self.phase in ["train", "validation"]:
            # 正常画像のみ (train/good)
            good_path = os.path.join(self.category_path, "train", "good")
            self.image_paths = sorted(glob.glob(os.path.join(good_path, "*.png")))
            # train/val では正常マスク (0 で埋めたマスク) を返す想定なので、実ファイルは None として保持
            self.mask_paths = [None] * len(self.image_paths)

        elif self.phase == "test":
            test_path = os.path.join(self.category_path, "test")
            # test ディレクトリ直下のサブフォルダを走査
            subfolders = sorted(os.listdir(test_path))
            for subfolder in subfolders:
                subfolder_path = os.path.join(test_path, subfolder)
                # サブフォルダ以下の PNG 画像をすべて集める
                image_files = sorted(glob.glob(os.path.join(subfolder_path, "*.png")))

                if subfolder == "good":
                    # 正常テスト画像
                    for img_path in image_files:
                        self.image_paths.append(img_path)
                        # 正常なのでマスクは None
                        self.mask_paths.append(None)
                else:
                    # 異常画像
                    gt_path = os.path.join(
                        self.category_path, "ground_truth", subfolder
                    )
                    for img_path in image_files:
                        filename = os.path.basename(img_path)
                        mask_file = os.path.join(
                            gt_path, filename.replace(".png", "_mask.png")
                        )
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_file)

        else:
            raise ValueError(
                f"Unknown phase: {self.phase}. Use 'train', 'validation', or 'test'."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 画像の読み込み
        image = Image.open(img_path).convert("RGB")

        # マスクが None (正常) ならば、同サイズのゼロ配列を生成する
        if mask_path is None:
            # (H, W) = (image.height, image.width)
            mask = Image.fromarray(
                np.zeros((image.height, image.width), dtype=np.uint8)
            )
        else:
            mask = Image.open(mask_path).convert("L")  # グレースケール

        # 画像変換を適用
        if self.transform:
            image = self.transform(image)

        # マスク変換を適用
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
