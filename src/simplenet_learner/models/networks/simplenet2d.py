from typing import Literal

import torch.nn as nn

from simplenet_learner.models.networks.backborn import ResnetFeatureExtractor
from simplenet_learner.models.networks.descriminator import Descriminator2D
from simplenet_learner.models.networks.projection import Projection2D


class Simplenet2D(nn.Module):
    def __init__(
        self,
        backborn_arch: Literal[
            "resnet18", "resnet50", "resnet101", "wide_resnet50_2", "wide_resnet101_2"
        ] = "resnet18",
        backborn_pretrained: bool = True,
        backborn_trainable: bool = False,
        projection_channel: int = 384,
        projection_layer_num: int = 1,
        descriminator_layer_num: int = 3,
        descriminator_reduce_rate: float = 1.5,
    ):
        super(Simplenet2D, self).__init__()
        self.backborn = ResnetFeatureExtractor(
            arch=backborn_arch, pretrained=backborn_pretrained
        )
        if not backborn_trainable:
            for param in self.backborn.parameters():
                param.requires_grad = False
            self.backborn.eval()

        self.projection = Projection2D(
            in_channel=self.get_backborn_channels(backborn_arch),
            out_channel=projection_channel,
            num_layers=projection_layer_num,
        )
        self.descriminator = Descriminator2D(
            in_channel=projection_channel,
            num_layer=descriminator_layer_num,
            reduce_rate=descriminator_reduce_rate,
        )

    def get_backborn_channels(
        self,
        arch: Literal[
            "resnet18", "resnet50", "resnet101", "wide_resnet50_2", "wide_resnet101_2"
        ],
    ) -> int:
        if arch in ["resnet18"]:
            return 384
        elif arch in ["resnet50", "resnet101"]:
            return 1536
        elif arch in ["wide_resnet50_2", "wide_resnet101_2"]:
            return 3072
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.backborn(x)
        x = self.projection(x)
        x = self.descriminator(x)
        return x

    def forward_features(self, x):
        x = self.backborn(x)
        x = self.projection(x)
        return x

    def forward_descriminator(self, x):
        x = self.descriminator(x)
        return x
