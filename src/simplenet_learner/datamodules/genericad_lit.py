from pathlib import Path
from typing import Optional, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torchvision import transforms

from simplenet_learner.datamodules.components.genericad import (
    GenericAnomalyDetectionDataset,
)
from simplenet_learner.datamodules.components.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class GenericMVTecLitDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        category: str,
        transform_cfg: DictConfig,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.category = category
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform_cfg = transform_cfg

        # 画像変換(例)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.transform_cfg.resize_h, self.transform_cfg.resize_w)
                ),
                transforms.RandomHorizontalFlip(self.transform_cfg.h_flip_p),
                transforms.RandomVerticalFlip(self.transform_cfg.v_flip_p),
                transforms.CenterCrop(
                    (self.transform_cfg.crop_h, self.transform_cfg.crop_w)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.transform_cfg.resize_h, self.transform_cfg.resize_w)
                ),
                transforms.CenterCrop(
                    (self.transform_cfg.crop_h, self.transform_cfg.crop_w)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        # マスク変換(例)
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.transform_cfg.resize_h, self.transform_cfg.resize_w)
                ),
                transforms.CenterCrop(
                    (self.transform_cfg.crop_h, self.transform_cfg.crop_w)
                ),
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self):
        # 必要に応じてダウンロード等をここで実行
        pass

    def setup(self, stage: Optional[str] = None):
        """
        stage == 'fit' or None: train/val用データセットを作る
        stage == 'test' or None: test用データセットを作る
        """
        if stage == "fit" or stage is None:
            full_dataset = GenericAnomalyDetectionDataset(
                root=self.data_dir,
                category=self.category,
                phase="train",
                transform=self.train_transform,  # 画像変換
                mask_transform=self.mask_transform,  # マスク変換（trainだと基本的に使わない）
            )
            n_samples = len(full_dataset)
            if n_samples == 0:
                raise ValueError(
                    f"No training data found under {self.data_dir}/{self.category}/train/good/"
                )
            train_size = int(n_samples * 0.8)
            val_size = n_samples - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            # test ディレクトリが無い、あるいは空の場合は FileNotFoundError が発生するよう実装している
            # ここで例外処理してもよいし、そのままエラーにしてもよい
            self.test_dataset = GenericAnomalyDetectionDataset(
                root=self.data_dir,
                category=self.category,
                phase="test",
                transform=self.test_transform,
                mask_transform=self.mask_transform,
            )

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            raise RuntimeError("train_dataset not found. Did you call setup('fit')?")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            raise RuntimeError("val_dataset not found. Did you call setup('fit')?")
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            raise RuntimeError("test_dataset not found. Did you call setup('test')?")
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
