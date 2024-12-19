from typing import Optional

from torch import nn


class Projection(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: Optional[int] = None, num_layers: int = 1
    ):
        super(Projection, self).__init__()

        if out_channels is None:
            out_channels = in_channels
        self.layers = nn.Sequential()
        for i in range(num_layers):
            if num_layers == 1:
                self.layers.add_module(
                    f"projection_{i}", nn.Linear(in_channels, out_channels)
                )
            else:
                (
                    self.layers.add_module(
                        f"projection_{i}",
                        nn.Linear(in_channels, out_channels),
                        nn.LeakyReLU(0.2),
                    ),
                )
