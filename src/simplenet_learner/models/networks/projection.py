from typing import Optional

from torch import nn


class Projection(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: Optional[int] = None, num_layers: int = 1
    ):
        super(Projection, self).__init__()

        if out_channel is None:
            out_channel = in_channel
        self.layers = nn.Sequential()
        for i in range(num_layers):
            if num_layers == 1:
                self.layers.add_module(
                    f"projection_{i}", nn.Linear(in_channel, out_channel)
                )
            else:
                self.layers.add_module(
                    f"projection_{i}",
                    nn.Sequential(
                        nn.Linear(in_channel, out_channel),
                        nn.LeakyReLU(0.2),
                    ),
                )

    def forward(self, x):
        return self.layers(x)


class Projection2D(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: Optional[int] = None, num_layers: int = 1
    ):
        super(Projection2D, self).__init__()

        if out_channel is None:
            out_channel = in_channel
        self.layers = nn.Sequential()
        for i in range(num_layers):
            if num_layers == 1:
                self.layers.add_module(
                    f"projection_{i}", nn.Conv2d(in_channel, out_channel, kernel_size=1)
                )
            else:
                self.layers.add_module(
                    f"projection_{i}",
                    nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size=1),
                        nn.LeakyReLU(0.2),
                    ),
                )

    def forward(self, x):
        return self.layers(x)


class SpatiallyIndependentProjection2D(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: Optional[int] = None, num_layers: int = 1
    ):
        super().__init__()

        if out_channel is None:
            out_channel = in_channel
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channel if i == 0 else out_channel
            out_ch = out_channel
            self.layers.append(nn.Linear(in_ch, out_ch))
            if i < num_layers - 1:
                self.layers.append(nn.LeakyReLU(0.2))

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        # 空間方向をフラット化して完全に独立に処理
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)  # [B*H*W, C]

        for layer in self.layers:
            x_flat = layer(x_flat)

        # 元の空間構造に戻す
        return x_flat.reshape(b, h, w, -1).permute(0, 3, 1, 2)  # [B, out_channel, H, W]
