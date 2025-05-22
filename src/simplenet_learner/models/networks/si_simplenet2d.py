from typing import Literal

import torch.nn as nn

from simplenet_learner.models.networks.backborn import ResnetFeatureExtractorSI
from simplenet_learner.models.networks.discriminator import (
    SpatiallyIndependentDiscriminator2D,
)
from simplenet_learner.models.networks.projection import (
    SpatiallyIndependentProjection2D,
)


class SpatiallyIndependentSimplenet2D(nn.Module):
    def __init__(
        self,
        backborn_arch: Literal[
            "resnet18", "resnet50", "resnet101", "wide_resnet50_2", "wide_resnet101_2"
        ] = "resnet18",
        backborn_pretrained: bool = True,
        backborn_trainable: bool = False,
        projection_channel: int = 384,
        projection_layer_num: int = 1,
        discriminator_layer_num: int = 3,
        discriminator_reduce_rate: float = 1.5,
        use_backborn_layers: Literal["1_2", "2_3", "1_2_3"] = "2_3",
        patch_size: int = 3,
    ):
        super(SpatiallyIndependentSimplenet2D, self).__init__()
        self.backborn = ResnetFeatureExtractorSI(
            arch=backborn_arch,
            pretrained=backborn_pretrained,
            projection_channel=projection_channel,
            use_layers=use_backborn_layers,
            patch_size=patch_size,
        )
        if not backborn_trainable:
            for param in self.backborn.parameters():
                param.requires_grad = False
            self.backborn.eval()

        self.projection = SpatiallyIndependentProjection2D(
            in_channel=projection_channel,
            out_channel=projection_channel,
            num_layers=projection_layer_num,
        )
        self.discriminator = SpatiallyIndependentDiscriminator2D(
            in_channel=projection_channel,
            num_layer=discriminator_layer_num,
            reduce_rate=discriminator_reduce_rate,
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
        x = self.discriminator(x)
        return x

    def forward_features(self, x):
        x = self.backborn(x)
        x = self.projection(x)
        return x

    def forward_discriminator(self, x):
        x = self.discriminator(x)
        return x
