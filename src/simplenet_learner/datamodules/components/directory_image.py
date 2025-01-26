import os
from typing import Optional, Callable
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class DirectoryImageDataset(Dataset):
    """単一のディレクトリから画像だけを読み込むDataset。
       ラベルは付与しない（返却は画像のみ）。
    """
    def __init__(self, 
                 folder_path: str, 
                 transform: Optional[Callable] = None,
                 extensions: tuple[str] = ('.png', '.jpg', '.jpeg', '.bmp')):
        """
        Args:
            folder_path (str): 画像ファイルが保存されているディレクトリ。
            transform (Callable, optional): 画像に適用する変換(前処理)。transforms.Compose など。
            extensions (tuple[str], optional): 対象とするファイル拡張子のタプル。
        """
        self.folder_path = folder_path
        self.transform = transform
        self.extensions = extensions

        # ディレクトリ内のファイル一覧を収集
        self.image_paths = []
        for f in os.listdir(folder_path):
            # 拡張子が対象ならリストに追加
            if f.lower().endswith(self.extensions):
                self.image_paths.append(os.path.join(folder_path, f))

        # ソートしておくと再現性が上がる
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # torchvision.io の read_image を使う場合 (Tensor, shape = C,H,W)
        # image = read_image(img_path)  # 0~255の範囲の整数型

        # Pillow (PIL) を使う場合
        with Image.open(img_path) as img:
            image = img.convert("RGB")  # 必要に応じてRGB変換

        if self.transform is not None:
            image = self.transform(image)

        return image
