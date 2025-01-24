from typing import Optional

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config/", config_name="train.yaml", version_base="1.1")
def main(config: DictConfig) -> Optional[float]:
    from simplenet_learner.training_pipeline import training_pipeline
    from simplenet_learner.utils import extras

    extras(config)

    return training_pipeline(config)


if __name__ == "__main__":
    main()
