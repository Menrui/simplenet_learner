from torch import nn

from simplenet_learner.models.networks import init_weight


class Descriminator(nn.Module):
    def __init__(self, in_channel: int, num_layer: int = 3, reduce_rate: float = 1):
        super(Descriminator, self).__init__()

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


class Descriminator2D(nn.Module):
    def __init__(self, in_channel: int, num_layer: int = 3, reduce_rate: float = 1):
        super(Descriminator2D, self).__init__()

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
