import torch


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) or (C, H, W) to be normalized.
        Returns:
            Tensor: DeNormalized image.
        """
        if tensor.ndimension() == 3:
            return self._denormalize(tensor)
        elif tensor.ndimension() == 4:
            tensor_list = []
            for t in tensor:
                tensor_list.append(self._denormalize(t))
            return torch.stack(tensor_list)
        else:
            raise ValueError(
                "UnNormalize only supports 3D or 4D tensors, but got tensor with dimension: "
                f"{tensor.ndimension()}"
            )

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: DeNormalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"
