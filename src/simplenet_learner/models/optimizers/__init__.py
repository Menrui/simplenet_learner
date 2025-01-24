from omegaconf import DictConfig
from torch import nn, optim


def get_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    if cfg.name == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.name == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {cfg.name} not implemented")
