from typing import Union

import numpy as np
import torch


def extract_patches_from_tensor(
    tensor: torch.Tensor,
    patch_size: int = 3,
    stride: int = 1,
    return_spatial_info: bool = False,
):
    """
    Extract patches from a given feature map tensor.

    This function takes a feature map (e.g., output from a convolutional backbone)
    of shape (B, C, H, W) and extracts overlapping patches of size `patch_size x patch_size`,
    spaced by `stride`. The result is a tensor with shape
    (B, N_patches, C, patch_size, patch_size), where N_patches depends on H, W,
    the patch size, and the stride.

    Args:
        tensor (torch.Tensor):
            Input tensor of shape (B, C, H, W), where:
                B: batch size
                C: number of channels
                H: height of the feature map
                W: width of the feature map
        patch_size (int):
            The size of each square patch (patch_size x patch_size).
        stride (int, optional):
            The stride (distance between patch centers). If None, stride is set to 1.
        return_spatial_info (bool, optional):
            If True, also returns the number of patches along each spatial dimension
            as (num_patches_height, num_patches_width).

    Returns:
        torch.Tensor or tuple:
            - If return_spatial_info is False, returns a tensor of shape
            (B, N_patches, C, patch_size, patch_size).
            - If return_spatial_info is True, returns a tuple:
            (patches, [num_patches_height, num_patches_width]).

    Example:
        >>> x = torch.randn(2, 64, 32, 32)  # B=2, C=64, H=32, W=32
        >>> patches = extract_patches_from_tensor(x, patch_size=3, stride=1)
        >>> print(patches.shape)
        # Might be something like (2, 900, 64, 3, 3) depending on patch configuration.
    """
    padding = int((patch_size - 1) / 2)
    unfolder = torch.nn.Unfold(
        kernel_size=patch_size, stride=stride, padding=padding, dilation=1
    )
    unfolded_features = unfolder(tensor)
    number_of_total_patches = []
    for s in tensor.shape[-2:]:
        n_patches = (s + 2 * padding - 1 * (patch_size - 1) - 1) / stride + 1
        number_of_total_patches.append(int(n_patches))
    unfolded_features = unfolded_features.reshape(
        *tensor.shape[:2], patch_size, patch_size, -1
    )
    unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

    if return_spatial_info:
        return unfolded_features, number_of_total_patches
    return unfolded_features


def rearrange_patches_to_batch(patches: torch.Tensor, batchsize: int):
    """
    Rearrange a flattened patches tensor back to a batch-major form.

    Some processing pipelines flatten both the batch dimension (B) and the
    patches dimension (N_patches) into a single dimension (B*N_patches) for
    convenience in computation. This function reshapes the tensor back into
    a form where batch and patches are separate dimensions, making it
    (B, N_patches, ...).

    For example, if you have a tensor of shape (B*N_patches, C, H', W'), this
    function can reshape it back to (B, N_patches, C, H', W') given the known B.

    Args:
        patches (torch.Tensor):
            Input tensor of shape (B*N_patches, ...).
        batch_size (int):
            The original batch size B.

    Returns:
        torch.Tensor:
            The reshaped tensor of shape (B, N_patches, ...).

    Example:
        >>> # Suppose we had a (2, 900, 64, 3, 3) tensor from patch extraction,
        >>> # and we processed it into shape (2*900, 64, 3, 3) = (1800, 64, 3, 3).
        >>> # After some computation, we want to go back to (2, 900, 64, 3, 3):
        >>> x = torch.randn(1800, 64, 3, 3)
        >>> x_rearranged = rearrange_patches_to_batch(x, batch_size=2)
        >>> print(x_rearranged.shape)  # (2, 900, 64, 3, 3)
    """

    return patches.reshape(batchsize, -1, *patches.shape[1:])


def compute_image_score_from_patches(patches: Union[torch.Tensor, np.ndarray], top_k=1):
    """
    Compute a single image-level anomaly score from patch-level scores or features.

    This function takes a high-dimensional tensor representing patch-level information
    (e.g., anomaly scores or features extracted from image patches) and progressively
    aggregates it into a single scalar score per image. The aggregation is done
    by taking the maximum (or top-k average) across the spatial and channel dimensions,
    ultimately reducing a (B, N_patches, C, H', W') tensor down to a (B,) tensor,
    where each element is the final anomaly score for one image.

    Args:
        x (torch.Tensor or np.ndarray):
            The input tensor of shape (B, N_patches, C, H', W'), where:
                - B: batch size (number of images)
                - N_patches: number of patches per image
                - C: number of channels or feature dimensions per patch
                - H', W': spatial dimensions of each patch
            The input may have fewer dimensions (e.g., (B, N_patches, C) or (B, N_patches))
            if some pooling was performed prior. The function reduces dimensions by repeatedly
            taking the max along the last dimension until a (B, N_patches) shape is reached.
        top_k (int, optional):
            If top_k > 1, the function takes the top k values along the N_patches dimension
            and averages them to produce the final score. If top_k <= 1, it simply takes the
            maximum patch score per image.

    Returns:
        torch.Tensor or np.ndarray:
            A one-dimensional tensor or array of shape (B,) containing a single score per image.
            The returned type matches the input type (torch.Tensor or np.ndarray).

    Example:
        >>> # Suppose x is a tensor of shape (2, 16, 64, 3, 3) corresponding to:
        >>> # B=2 images, N_patches=16, C=64 channels, and patch size H'=W'=3
        >>> # compute_image_score_from_patches will reduce it to (2,) via max pooling.
        >>> scores = compute_image_score_from_patches(x)
        >>> print(scores.shape)  # torch.Size([2])
    """
    was_numpy = False
    if isinstance(patches, np.ndarray):
        was_numpy = True
        patches = torch.from_numpy(patches)
    while patches.ndim > 2:
        patches = torch.max(patches, dim=-1).values
    if patches.ndim == 2:
        if top_k > 1:
            image_scores = torch.topk(patches, top_k, dim=1).values.mean(1)
        else:
            image_scores = torch.max(patches, dim=1).values
    if was_numpy:
        return image_scores.numpy()
    return image_scores
