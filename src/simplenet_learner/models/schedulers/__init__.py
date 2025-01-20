from omegaconf import Dictonfig
from torch import optim


def get_lr_sheduler(optimizer, cfg: Dictonfig) -> optim.lr_scheduler.LRScheduler:
    if cfg.scheduler.name == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma
        )
    elif cfg.scheduler.name == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma
        )
    elif cfg.scheduler.name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler.mode,
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
        )
    elif cfg.scheduler.name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.scheduler.T_max, eta_min=cfg.scheduler.eta_min
        )
    else:
        raise ValueError(f"Scheduler {cfg.scheduler.name} not implemented")
