from torch import nn

from src.simplenet_learner.models.components.init_weight import init_weight


class Descriminator(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, reduce_rate: int = 2):
        super(Descriminator, self).__init__()

        self.layers = nn.Sequential()
        for i in range(num_layers - 1):
            _in_layer_channel = (
                in_channels if i == 0 else in_channels // (reduce_rate**i)
            )
            self.layers.add_module(
                f"linear_block_{i}",
                nn.Sequential(
                    nn.Linear(_in_layer_channel, _in_layer_channel // reduce_rate),
                    nn.BatchNorm1d(_in_layer_channel // reduce_rate),
                    nn.LeakyReLU(0.2),
                ),
            )
        self.output = nn.Linear(
            in_channels // (reduce_rate ** (num_layers - 1)), 1, bias=False
        )
        self.apply(init_weight)

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
        return x
