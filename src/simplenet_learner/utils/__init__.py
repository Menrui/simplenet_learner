import random
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def save_checkpoint(
    state: Dict,
    is_best: bool,
    save_dir: Path = Path("checkpoints"),
    filename: str = "checkpoint.pth",
):
    torch.save(state, str(save_dir.joinpath(filename)))
    if is_best:
        shutil.copyfile(
            str(save_dir.joinpath(filename)), str(save_dir.joinpath("best.pth"))
        )
