import torch
from torch import nn
from torch.nn import functional as F


class ChannelPooler(nn.Module):
    def __init__(self, out_channels=384):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力テンソル x に対してチャネル方向の平均プーリングを適用します。
        この操作は、チャネル数を out_channels に変換します。

        Args:
            x (torch.Tensor): 入力テンソル。形状は [B, C, H, W]。

        Returns:
            torch.Tensor: 出力テンソル。形状は [B, out_C, H, W]。
        """
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # チャネル方向をプーリング用の最後の次元に
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)

        # adaptive_avg_pool1dでチャネル数を変換
        x_pooled = F.adaptive_avg_pool1d(x_reshaped, self.out_channels)

        # 元の形状に戻す
        x_result = x_pooled.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)

        return x_result
