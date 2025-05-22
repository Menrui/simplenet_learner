from torch import nn

from simplenet_learner.models.networks import init_weight


class Discriminator(nn.Module):
    def __init__(self, in_channel: int, num_layer: int = 3, reduce_rate: float = 1):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential()
        for i in range(num_layer - 1):
            _in_layer_channel = (
                in_channel if i == 0 else int(in_channel // (reduce_rate**i))
            )
            self.layers.add_module(
                f"linear_block_{i}",
                nn.Sequential(
                    nn.Linear(_in_layer_channel, int(_in_layer_channel // reduce_rate)),
                    nn.BatchNorm1d(int(_in_layer_channel // reduce_rate)),
                    nn.LeakyReLU(0.2),
                ),
            )
        self.output = nn.Linear(
            int(in_channel // (reduce_rate ** (num_layer - 1))), 1, bias=False
        )
        self.apply(init_weight)

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
        return x


class Discriminator2D(nn.Module):
    def __init__(self, in_channel: int, num_layer: int = 3, reduce_rate: float = 1):
        super(Discriminator2D, self).__init__()

        self.layers = nn.Sequential()
        for i in range(num_layer - 1):
            _in_layer_channel = (
                in_channel if i == 0 else int(in_channel // (reduce_rate**i))
            )
            self.layers.add_module(
                f"conv_block_{i}",
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=_in_layer_channel,
                        out_channels=int(_in_layer_channel // reduce_rate),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                    nn.BatchNorm2d(int(_in_layer_channel // reduce_rate)),
                    nn.LeakyReLU(0.2),
                ),
            )
        self.output = nn.Conv2d(
            in_channels=int(in_channel // (reduce_rate ** (num_layer - 1))),
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.apply(init_weight)

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
        return x


class SpatiallyIndependentDiscriminator2D(nn.Module):
    def __init__(self, in_channel: int, num_layer: int = 3, reduce_rate: float = 1):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layer - 1):
            _in_layer_channel = (
                in_channel if i == 0 else int(in_channel // (reduce_rate**i))
            )
            _out_layer_channel = int(_in_layer_channel // reduce_rate)

            self.layers.append(nn.Linear(_in_layer_channel, _out_layer_channel))
            self.layers.append(nn.BatchNorm1d(_out_layer_channel))
            self.layers.append(nn.LeakyReLU(0.2))

        self.output = nn.Linear(
            # int(in_channel // (reduce_rate ** (num_layer - 1))), 1, bias=False
            _out_layer_channel,
            1,
            bias=False,
        )
        self.apply(init_weight)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        # 空間方向をフラット化して完全に独立に処理
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)  # [B*H*W, C]

        for i in range(0, len(self.layers), 3):
            linear = self.layers[i]
            bn = self.layers[i + 1]
            relu = self.layers[i + 2]

            x_flat = linear(x_flat)
            # BatchNorm1dは各位置で独立に正規化
            x_flat = bn(x_flat)
            x_flat = relu(x_flat)

        x_flat = self.output(x_flat)

        # 元の空間構造に戻す
        return x_flat.reshape(b, h, w, -1).permute(0, 3, 1, 2)  # [B, 1, H, W]
