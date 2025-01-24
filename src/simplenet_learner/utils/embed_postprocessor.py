from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


class RescaleSegmentator:
    """
    A class used to rescale and process segmentation maps from patch scores and features.
    Attributes:
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').
        target_size (int or tuple): The target size for interpolation. Default is 224.
        smoothing (int): The sigma value for Gaussian filtering. Default is 4.
    Methods:
        convert_to_segmentation(patch_scores, features):
            Convert patch scores and features to segmentation maps using interpolation and Gaussian filtering.
    """

    def __init__(
        self, device: torch.device, target_size: Union[int, tuple[int, int]] = 224
    ):
        self.device = device
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(
        self,
        patch_scores: Union[torch.Tensor, np.ndarray],
        features: torch.Tensor,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Convert patch scores and features to segmentation maps.
        This method takes patch scores and features, processes them using
        interpolation and Gaussian filtering, and returns the processed
        segmentation maps.
        Args:
            patch_scores (np.ndarray or torch.Tensor): The patch scores to be
                converted. Can be a NumPy array or a PyTorch tensor.
            features (np.ndarray or torch.Tensor): The features to be converted.
                Can be a NumPy array or a PyTorch tensor.
        Returns:
            tuple: A tuple containing:
                - List of np.ndarray: The processed patch scores after applying
                  Gaussian filtering.
                - List of np.ndarray: The processed features after interpolation.
        """

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                input=_scores,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features)
            features = features.to(self.device).permute(0, 3, 1, 2)
            if (
                self.target_size[0]
                * self.target_size[1]
                * features.shape[0]
                * features.shape[1]
                >= 2**31
            ):
                subbatch_size = int(
                    (2**31 - 1)
                    / (self.target_size[0] * self.target_size[1] * features.shape[1])
                )
                _interpolated_features = []
                for i_subbatch in range(int(features.shape[0] / subbatch_size + 1)):
                    subfeatures = features[
                        i_subbatch * subbatch_size : (i_subbatch + 1) * subbatch_size
                    ]
                    subfeatures = (
                        subfeatures.unsqueeze(0)
                        if len(subfeatures.shape) == 3
                        else subfeatures
                    )
                    subfeatures = F.interpolate(
                        subfeatures,
                        size=self.target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    _interpolated_features.append(subfeatures)
                features = torch.cat(_interpolated_features, 0)
            else:
                features = F.interpolate(
                    features,
                    size=self.target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            interpolated_features = features.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ], [feature for feature in interpolated_features]
