import torch
from torch import nn


class PatchExtractor(nn.Module):
    def __init__(self, patch_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        入力テンソルからパッチを抽出し、空間情報を保持したまま変換します。
        Args:
            - x (torch.Tensor): 入力テンソル。形状は [バッチサイズ, チャンネル数, 高さ, 幅]。
        Returns:
            - patches (torch.Tensor): パッチ化されたテンソル。形状は [バッチサイズ, パッチ数, パッチごとの特徴量次元]。
            - spatial_size (int, int): パッチ化後の空間サイズ (高さ, 幅)。
        """
        # パッチ抽出（空間関係を保持）
        B, C, H, W = x.shape
        patches = self.unfold(x)  # [B, C*patch_size*patch_size, L]
        # L = (H - patch_size) / stride + 1
        # stride = patch_size なので、L = (H - patch_size) / patch_size + 1
        # L = (H // patch_size) * (W // patch_size)

        # 空間情報の記録と再構成可能な形式に変換
        patches = patches.transpose(1, 2)  # [B, L, C*patch_size*patch_size]
        return patches, (H // self.patch_size, W // self.patch_size)


class PatchesToFeatureMap(nn.Module):
    def forward(
        self, patches: torch.Tensor, spatial_shape: tuple[int, int]
    ) -> torch.Tensor:
        # [B, L, C*patch_size*patch_size] → [B, C*patch_size*patch_size, H, W]
        B, L, patch_dim = patches.shape
        H, W = spatial_shape
        return patches.reshape(B, H, W, patch_dim).permute(0, 3, 1, 2)


# class SpatiallyIndependentPreprocessor2D(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         # より単純な実装
#         self.bn = nn.BatchNorm1d(out_channels)

#     def forward(self, x: torch.Tensor):
#         """
#         入力テンソル x に対して畳み込みと空間的に独立した正規化を適用します。

#         Args:
#             x (torch.Tensor): 入力テンソル。形状は [B, C, H, W]。

#         Returns:
#             torch.Tensor: 出力テンソル。形状は [B, out_C, H, W]。
#         """
#         # x: [B, C, H, W]
#         B, C, H, W = x.shape
#         x = self.conv(x)  # [B, out_C, H, W]

#         # 空間位置ごとに独立して正規化
#         x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))  # [B*H*W, out_C]
#         x_normed = self.bn(x_flat)  # [B*H*W, out_C]
#         x = x_normed.reshape(B, H, W, x.size(1)).permute(0, 3, 1, 2)  # [B, out_C, H, W]

#         return x


# class SpatiallyIndependentAggregator2D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         # 空間位置ごとに独立した処理
#         return self.conv(x)
