from typing import Optional

from omegaconf import DictConfig
from torch import optim


def get_lr_scheduler(
    optimizer: optim.Optimizer, cfg: Optional[DictConfig]
) -> Optional[optim.lr_scheduler.LRScheduler]:
    if cfg is None:
        return None
    if cfg.name == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma
        )
    elif cfg.name == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.milestones, gamma=cfg.gamma
        )
    elif cfg.name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.mode,
            factor=cfg.factor,
            patience=cfg.patience,
        )
    elif cfg.name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.T_max, eta_min=cfg.eta_min
        )
    else:
        raise ValueError(f"Scheduler {cfg.name} not implemented")
