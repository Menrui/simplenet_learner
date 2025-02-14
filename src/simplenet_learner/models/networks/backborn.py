from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from typing import Literal
else:
    try:
        from typing import Literal
    except ImportError:
        from typing_extensions import Literal

from abc import ABC, abstractmethod

from torch import fx
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    resnet18,
    resnet50,
    resnet101,
    wide_resnet50_2,
    wide_resnet101_2,
)
from torchvision.models.feature_extraction import create_feature_extractor


class Backborn(nn.Module, ABC):
    @abstractmethod
    def get_feature_extraction_model(
        self, layers_dict: dict[str, str]
    ) -> fx.GraphModule:
        pass


class Resnet(Backborn):
    def __init__(
        self,
        arch: Literal[
            "resnet18", "resnet50", "resnet101", "wide_resnet50_2", "wide_resnet101_2"
        ],
        pretrain: bool = True,
    ) -> None:
        super().__init__()
        self.model = {
            "resnet18": resnet18(
                weights=ResNet18_Weights.DEFAULT if pretrain is True else None,
                progress=True,
            ),
            "resnet50": resnet50(weights=ResNet50_Weights.DEFAULT, progress=True),
            "resnet101": resnet101(weights=ResNet101_Weights.DEFAULT, progress=True),
            "wide_resnet50_2": wide_resnet50_2(
                weights=Wide_ResNet50_2_Weights.DEFAULT, progress=True
            ),
            "wide_resnet101_2": wide_resnet101_2(
                weights=Wide_ResNet101_2_Weights.DEFAULT, progress=True
            ),
        }.get(
            arch,
            resnet18(ResNet18_Weights.DEFAULT, progress=True),
        )

    def get_feature_extraction_model(
        self, layers_dict: dict[str, str]
    ) -> fx.GraphModule:
        """
        Creates a feature extraction model based on the provided layers dictionary.
        Args:
            layers_dict (dict[str, str]): A dictionary where keys are layer names
                                          and values are the corresponding output names.
        Returns:
            fx.GraphModule: A feature extraction model.

        Example:
            >>> layers_dict = {"layer1": "output1", "layer2": "output2"}
            >>> feature_extractor = self.get_feature_extraction_model(layers_dict)
        """

        return create_feature_extractor(self.model, layers_dict)


class ResnetFeatureExtractor(nn.Module):
    def __init__(
        self,
        arch: Literal[
            "resnet18", "resnet50", "resnet101", "wide_resnet50_2", "wide_resnet101_2"
        ],
        pretrained: bool = True,
        pool_kernel_size: int = 3,
        pool_stride: int = 1,
    ) -> None:
        super().__init__()

        if arch == "resnet18":
            base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet50":
            base = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet101":
            base = resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None)
        elif arch == "wide_resnet50_2":
            base = wide_resnet50_2(
                weights=Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            )
        elif arch == "wide_resnet101_2":
            base = wide_resnet101_2(
                weights=Wide_ResNet101_2_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.conv1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)

        if l2.shape[-2:] != l3.shape[-2:]:
            l3 = nn.functional.interpolate(l3, size=l2.shape[-2:], mode="bilinear")
        out = torch.cat([l2, l3], dim=1)

        out = nn.functional.avg_pool2d(
            out, kernel_size=self.pool_kernel_size, stride=self.pool_stride, padding=1
        )

        return out
