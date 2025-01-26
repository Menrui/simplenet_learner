from typing import TYPE_CHECKING

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
    resnet18,
    resnet50,
    resnet101,
    wide_resnet50_2,
    wide_resnet101_2, 
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights
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
            "resnet18": resnet18(weights=ResNet18_Weights.DEFAULT if pretrain is True else None, progress=True),
            "resnet50": resnet50(weights=ResNet50_Weights.DEFAULT, progress=True),
            "resnet101": resnet101(weights=ResNet101_Weights.DEFAULT, progress=True),
            "wide_resnet50_2": wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT, progress=True),
            "wide_resnet101_2": wide_resnet101_2(weights=Wide_ResNet101_2_Weights.DEFAULT, progress=True),
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
