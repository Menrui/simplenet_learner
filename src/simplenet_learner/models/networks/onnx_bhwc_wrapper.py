import torch
from torch import nn


class OnnxBhwcWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x_bhwc: torch.Tensor):
        """
        x_bhwc: (B, H, W, C) という形で受け取り、
                 (B, C, H, W) に permute して base_model に渡す
        """
        # (B, H, W, C) => (B, C, H, W)
        x_bchw = x_bhwc.permute(0, 3, 1, 2)
        out_bchw = self.base_model(x_bchw)
        # 必要に応じて (B, C, H, W) => (B, H, W, C) に戻すならここで permute
        # 今回は出力も BHWC にしたいなら:
        out_bhwc = out_bchw.permute(0, 2, 3, 1)
        return out_bhwc
        # return out_bchw  # こっちだと出力が BCHW のままになる
