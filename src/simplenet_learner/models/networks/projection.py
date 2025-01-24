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
                (
                    self.layers.add_module(
                        f"projection_{i}",
                        nn.Linear(in_channel, out_channel),
                        nn.LeakyReLU(0.2),
                    ),
                )

    def forward(self, x):
        return self.layers(x)
