from typing import TYPE_CHECKING

import torch
from torch import nn

from simplenet_learner.models.networks.components.spatially_independent_processor import (
    PatchesToFeatureMap,
    PatchExtractor,
)

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


class ResnetFeatureExtractorSI(nn.Module):
    def __init__(
        self,
        arch: Literal[
            "resnet18", "resnet50", "resnet101", "wide_resnet50_2", "wide_resnet101_2"
        ],
        pretrained: bool = True,
        patch_size: int = 3,
        projection_channel: int = 384,
        use_layers: Literal["1_2", "2_3", "1_2_3"] = "1_2",
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

        self.patch_size = patch_size

        # for spatially independent processing module
        self.use_layers = use_layers
        # self.projection_channel_pooler = ChannelPooler(out_channels=projection_channel)
        l1_channel, l2_channel, l3_channel = self.get_backborn_channels(arch)
        projection_in_channel = {
            "1_2": l1_channel * patch_size**2 + l2_channel * patch_size**2,
            "2_3": l2_channel * patch_size**2 + l3_channel * patch_size**2,
            "1_2_3": l1_channel * patch_size**2
            + l2_channel * patch_size**2
            + l3_channel * patch_size**2,
        }
        self.projection_channel_pooler = nn.Conv2d(
            in_channels=projection_in_channel.get(use_layers, None),
            out_channels=projection_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.patch_extractor = PatchExtractor(patch_size=patch_size)
        self.patches_to_feature_map = PatchesToFeatureMap()

    def forward(self, x):
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)

        if l1.shape[-2:] != l2.shape[-2:] and "1" in self.use_layers:
            l2 = nn.functional.interpolate(l2, size=l1.shape[-2:], mode="bilinear")
        if l2.shape[-2:] != l3.shape[-2:] and "3" in self.use_layers:
            l3 = nn.functional.interpolate(l3, size=l2.shape[-2:], mode="bilinear")
        # out = torch.cat([l2, l3], dim=1)

        # [B, C*patch_size*patch_size, H, W]
        if "1" in self.use_layers:
            l1_patches = self._spatially_independent_postprocess(l1)
        l2_patches = self._spatially_independent_postprocess(l2)
        if "3" in self.use_layers:
            l3_patches = self._spatially_independent_postprocess(l3)

        # print("feature: ", l1.shape)
        # print("l1_patches: ", l1_patches.shape)
        # print("l2_patches: ", l2_patches.shape)
        # print("l3_patches: ", l3_patches.shape)
        layer_patches_list = []
        if "1" in self.use_layers:
            layer_patches_list.append(l1_patches)
        if "2" in self.use_layers:
            layer_patches_list.append(l2_patches)
        if "3" in self.use_layers:
            layer_patches_list.append(l3_patches)
        out = self.projection_channel_pooler(
            torch.cat(
                layer_patches_list,
                dim=1,
            )
        )
        # print("concat_pooler: ", out.shape)

        return out

    def _spatially_independent_postprocess(self, x):
        # patches: [B, L, C*patch_size*patch_size]
        # spatial_size: (H, W)
        patches, spatial_size = self.patch_extractor(x)
        # x: [B, C*patch_size*patch_size, H, W]
        x = self.patches_to_feature_map(patches, spatial_size)
        return x

    def get_backborn_channels(
        self,
        arch: Literal[
            "resnet18", "resnet50", "resnet101", "wide_resnet50_2", "wide_resnet101_2"
        ],
    ) -> tuple[int, int, int]:
        if arch in ["resnet18"]:
            return (64, 128, 256)
        elif arch in ["resnet50", "resnet101"]:
            return (256, 512, 1024)
        elif arch in ["wide_resnet50_2", "wide_resnet101_2"]:
            return (512, 1024, 2048)
        else:
            raise NotImplementedError
