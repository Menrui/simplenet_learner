import torch


class NoiseGenerator:
    def __init__(
        self,
        noise_std: float = 0.05,
        noise_type: str = "normal",
        noise_scaling: float = 1.1,
        num_clsses: int = 1,
    ):
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.noise_scaling = noise_scaling
        self.num_clsses = num_clsses

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate noise for the input tensor x.

        Args:
            x (torch.Tensor):
                The input tensor for which noise should be generated.

        Returns:
            torch.Tensor:
                The generated noise tensor.

        """
        noise_idxs = torch.randint(0, self.num_clsses, (x.size(0),), device=x.device)
        noise_one_hot = torch.nn.functional.one_hot(
            noise_idxs, num_classes=self.num_clsses
        ).to(x.device)
        if self.noise_type == "normal":
            noise = torch.stack(
                [
                    torch.normal(
                        mean=0,
                        std=self.noise_std * self.noise_scaling**i,
                        size=x.size(),
                        device=x.device,
                    )
                    for i in range(self.num_clsses)
                ],
                dim=1,
            ).to(x.device)
        else:
            raise ValueError(f"Unknown noise type {self.noise_type}")
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        return noise

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the input tensor x.

        Args:
            x (torch.Tensor):
                The input tensor to which noise should be added.

        Returns:
            torch.Tensor:
                The tensor with added noise.

        """
        noise = self(x)
        return x + noise


class NoiseGenerator2D:
    def __init__(
        self,
        noise_std: float = 0.05,
        noise_type: str = "normal",
        noise_scaling: float = 1.1,
        num_clsses: int = 1,  # クラス数
    ):
        """
        2次元入力 x: (B, H, W) に対して、学習クラスIDをもとにノイズを付与するジェネレータ。
        """
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.noise_scaling = noise_scaling
        self.num_clsses = num_clsses

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x の形状: (B, H, W)

        Returns:
            noise: x と同じ形状 (B, H, W) のノイズ
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3D: (B, H, W). Got: {x.shape}")

        # (B,) のランダムなクラスIDを生成
        noise_idxs = torch.randint(
            low=0, high=self.num_clsses, size=(x.size(0),), device=x.device
        )
        # (B, num_clsses) の One-hot
        noise_one_hot = torch.nn.functional.one_hot(
            noise_idxs, num_classes=self.num_clsses
        ).to(x.device)

        if self.noise_type == "normal":
            # それぞれのクラス用のノイズをスタック
            #  => shape: (B, num_clsses, H, W)
            noise = torch.stack(
                [
                    torch.normal(
                        mean=0,
                        std=self.noise_std * (self.noise_scaling**i),
                        size=x.shape,  # (B, H, W)
                        device=x.device,
                    )
                    for i in range(self.num_clsses)
                ],
                dim=1,
            )  # stacking along dimension=1
            # ここで noise の shape は (B, num_clsses, H, W)
        else:
            raise ValueError(f"Unknown noise type {self.noise_type}")

        # (B, num_clsses) の One-hot を、(B, num_clsses, 1, 1) にしてブロードキャスト
        # => (B, num_clsses, H, W) と乗算可能
        noise_one_hot = noise_one_hot.unsqueeze(-1).unsqueeze(-1)

        # 選択されたクラスIDに対応するノイズだけを残して合計
        # => shape: (B, H, W)
        noise = (noise * noise_one_hot).sum(dim=1)

        return noise

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成したノイズを x に加えて返す
        """
        noise = self.__call__(x)
        return x + noise
