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
        num_clsses: int = 1,
    ):
        """
        Args:
            noise_std (float): ノイズの標準偏差の初期値 (クラス0用)
            noise_type (str): ノイズの種類 ("normal" のみ実装)
            noise_scaling (float): クラスIDに応じたノイズ強度の倍率
            num_clsses (int): クラス数
        """
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.noise_scaling = noise_scaling
        self.num_clsses = num_clsses

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力テンソル x (B, C, H, W) と同じ形状のノイズを生成します。

        クラスIDをバッチ単位でランダムに決定し、ノイズを重み付けして返す。

        Args:
            x (torch.Tensor): 形状 (B, C, H, W)

        Returns:
            torch.Tensor: 生成されたノイズ (B, C, H, W)
        """
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D (B, C, H, W). Got: {x.shape}")

        B, C, H, W = x.shape

        # バッチごとにクラスIDをランダムに割り当て (shape: (B,))
        noise_idxs = torch.randint(0, self.num_clsses, (B,), device=x.device)

        # (B, num_clsses) のワンホット表現
        noise_one_hot = torch.nn.functional.one_hot(
            noise_idxs, num_classes=self.num_clsses
        ).to(x.device)

        # 各クラス用のノイズをスタック
        #  => shape: (B, num_clsses, C, H, W)
        if self.noise_type == "normal":
            noise = torch.stack(
                [
                    torch.normal(
                        mean=0.0,
                        std=self.noise_std * (self.noise_scaling**i),
                        size=x.size(),  # (B, C, H, W)
                        device=x.device,
                    )
                    for i in range(self.num_clsses)
                ],
                dim=1,
            )  # stacking along dim=1
        else:
            raise ValueError(f"Unknown noise type '{self.noise_type}'")

        # noise_one_hot の shape: (B, num_clsses)
        # => (B, num_clsses, 1, 1, 1) に拡張してブロードキャスト
        noise_one_hot = noise_one_hot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # => shape: (B, num_clsses, 1, 1, 1)

        # 要素ごとにクラスIDに該当するノイズだけを残して合計
        # => shape: (B, C, H, W)
        noise = (noise * noise_one_hot).sum(dim=1)

        return noise

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力に生成したノイズを加えて返す。

        Args:
            x (torch.Tensor): 形状 (B, C, H, W)

        Returns:
            torch.Tensor: x + noise
        """
        noise = self(x)  # (B, C, H, W)
        return x + noise


class SpatiallyIndependentNoiseGenerator2D:
    def __init__(
        self,
        noise_std=0.05,
        noise_type="normal",
        noise_scaling=1.1,
        num_clsses=1,
    ):
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.noise_scaling = noise_scaling
        self.num_clsses = num_clsses

    def __call__(self, x):
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D (B, C, H, W). Got: {x.shape}")

        B, C, H, W = x.shape

        # 各位置ごとに独立したクラスIDを生成
        noise_idxs = torch.randint(0, self.num_clsses, (B, H, W), device=x.device)
        noise_one_hot = torch.nn.functional.one_hot(
            noise_idxs, num_classes=self.num_clsses
        ).to(x.device)  # [B, H, W, num_clsses]

        if self.noise_type == "normal":
            # 各位置、各クラスに対して独立したノイズを生成
            noise = torch.stack(
                [
                    torch.normal(
                        mean=0.0,
                        std=self.noise_std * (self.noise_scaling**i),
                        size=(B, C, H, W),
                        device=x.device,
                    )
                    for i in range(self.num_clsses)
                ],
                dim=-1,
            )  # [B, C, H, W, num_clsses]
        else:
            raise ValueError(f"Unknown noise type '{self.noise_type}'")

        # 位置ごとのクラスIDを適用
        noise_one_hot = noise_one_hot.unsqueeze(1)  # [B, 1, H, W, num_clsses]
        noise = (noise * noise_one_hot).sum(dim=-1)  # [B, C, H, W]

        return noise

    def add_noise(self, x):
        noise = self(x)
        return x + noise
