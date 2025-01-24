from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset


class GenericAnomalyDetectionDataset(Dataset):
    """
    MVTec AD 風のディレクトリ構成を想定した汎用データセットクラス。

    想定ディレクトリ構成 (例):
        /path/to/...
        └─ category/
           ├─ train/
           │  └─ good/
           │     ├─ ***.png
           │     └─ ...
           ├─ test/
           │  ├─ good/
           │  │  ├─ ***.png
           │  │  └─ ...
           │  ├─ ano_class1/
           │  │  ├─ ***.png
           │  │  └─ ...
           │  └─ ano_class2/
           │     ├─ ***.png
           │     └─ ...
           └─ gt_mask/
              ├─ ano_class1/
              │  ├─ ***.png
              │  └─ ...
              └─ ano_class2/
                 ├─ ***.png
                 └─ ...

    - phase="train" の場合
        train/good/ 以下のすべての画像を読み込む。
        マスクは常に None を返す (良品なので)。
    - phase="test" の場合
        test/ 以下のサブフォルダを走査する。
        サブフォルダ名が "good" の場合は良品、
        それ以外は異常クラスとみなし、該当する gt_mask/サブフォルダ名/ を探す。
        画像ファイルと同名のマスクファイルがあれば読み込み、無ければ None を返す。

    ディレクトリが存在しない場合や、マスクが見つからない場合の動作は
    必要に応じて調整し、例外を投げる/Noneを返す等を使い分ける。
    """

    def __init__(
        self,
        root: Union[str, Path],
        category: str,
        phase: str = "train",
        transform=None,
        mask_transform=None,
    ):
        super().__init__()
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.category = category
        self.phase = phase
        self.transform = transform
        self.mask_transform = mask_transform

        # イメージとマスクのパスを保持するリスト
        self.image_paths: List[Path] = []
        self.mask_paths: List[Optional[Path]] = []

        self._collect_paths()

    def _collect_paths(self) -> None:
        """
        ディレクトリ構成に基づいて、self.image_paths と self.mask_paths を構築する。
        """
        category_path = self.root / self.category
        phase = self.phase.lower()

        if phase == "train":
            train_good_dir = category_path / "train" / "good"
            if not train_good_dir.is_dir():
                # train/good/ ディレクトリが存在しない場合はエラーにするか、空のままにするかは設計次第
                raise FileNotFoundError(f"Train directory not found: {train_good_dir}")
            # すべてのPNG画像をリストアップ
            self.image_paths = sorted(train_good_dir.glob("*.png"))
            # 全部良品なのでマスクは None
            self.mask_paths = [None] * len(self.image_paths)

        elif phase == "test":
            test_dir = category_path / "test"
            if not test_dir.is_dir():
                # test ディレクトリが存在しない場合
                # データセットとして扱えないのでエラーとする (または空のデータセットとする)
                raise FileNotFoundError(f"Test directory not found: {test_dir}")

            subfolders = [p for p in test_dir.iterdir() if p.is_dir()]
            subfolders = sorted(subfolders)  # good, ano_class1, ano_class2, etc.

            gt_mask_dir = category_path / "gt_mask"

            for subfolder in subfolders:
                image_files = sorted(subfolder.glob("*.png"))
                if subfolder.name == "good":
                    # 良品はマスク = None
                    for img_path in image_files:
                        self.image_paths.append(img_path)
                        self.mask_paths.append(None)
                else:
                    # 異常クラス
                    # gt_mask/サブフォルダ名 があればそこからマスクを探す
                    mask_subdir = gt_mask_dir / subfolder.name
                    for img_path in image_files:
                        self.image_paths.append(img_path)
                        if mask_subdir.is_dir():
                            # 画像ファイル名と同名のマスクがあれば使う
                            mask_path = mask_subdir / img_path.name
                            if mask_path.is_file():
                                self.mask_paths.append(mask_path)
                            else:
                                self.mask_paths.append(None)
                        else:
                            self.mask_paths.append(None)
        else:
            raise ValueError(f"Unknown phase: {phase}. Choose from 'train' or 'test'.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (image, mask)
            - image: shape = (C, H, W)
            - mask: shape = (1, H, W) あるいは (C, H, W)でもよいが、多くの場合グレイスケールで1ch
                     マスクが無い場合はゼロ埋めなどで代用する。
        """
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 画像読み込み
        image = Image.open(img_path).convert("RGB")

        # マスクがなければ、画像サイズに応じたゼロマスクを生成
        if mask_path is None:
            mask = Image.new("L", image.size, color=0)
        else:
            mask = Image.open(mask_path).convert("L")

        # 変換を適用
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            # mask_transform 未指定の場合は Tensor に変換だけ行う例 (お好みで)
            mask = torch.from_numpy(
                torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes())).numpy()
            ).view(mask.height, mask.width)
            mask = mask.float() / 255.0
            mask = mask.unsqueeze(0)  # (1, H, W)

        return image, mask
