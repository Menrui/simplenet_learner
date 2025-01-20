from omegaconf import DictConfig
from torch import nn, optim


def get_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    if cfg.optimizer.name == "Adam":
        return optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    elif cfg.optimizer.name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {cfg.optimizer.name} not implemented")
