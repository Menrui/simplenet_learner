import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class History(object):
    def __init__(
        self,
        keys: Sequence,
        output_dir: Union[Path, str],
        logger: Optional[logging.Logger] = None,
    ):
        self.output_dir = (
            output_dir if isinstance(output_dir, Path) else Path(output_dir)
        )
        self.keys = keys

        self.logs: Dict[str, Sequence] = {key: [] for key in keys}
        self.logger = logger if logger is not None else None

    def __call__(self, data):
        for key, value in data.items():
            self.logs[key].append(value)

    def save(self, filename: str = "history.pkl"):
        savepath = self.output_dir.joinpath(filename)
        with open(savepath, "wb") as f:
            pickle.dump(self.logs, f)

    def save_csv(self, filename: str = "history.csv"):
        pass

    def plot_graph(
        self,
        keys: Union[list, str],
        filename: str = "loss.png",
        xlabel: str = "Iteration",
        ylabel: str = "Loss",
        title: str = "Training curve.",
    ):
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # for i, (key, value) in enumerate(self.logs.items()):
        if isinstance(keys, list):
            max_elements = max([len(self.logs[k]) for k in keys])
            for i, key in enumerate(keys):
                value = self.logs[key]
                x = (
                    np.arange(len(value))
                    if not max_elements > len(value)
                    else np.linspace(0, max_elements, len(value))
                )
                y = np.array(value)
                ax.plot(x, y, label=key, color=cm.cividis(i / len(keys)))
        else:
            key = keys
            value = self.logs[key]
            x = np.arange(len(value))
            y = np.array(value)
            ax.plot(x, y, label=key)

        ax.legend(loc="best")

        save_path = str(Path(self.output_dir).joinpath(filename))
        if self.logger is not None:
            self.logger.info("Save {}".format(save_path))
        else:
            print("Save {}".format(save_path))
        plt.savefig(save_path)
        plt.clf()
        plt.cla()
        plt.close("all")
