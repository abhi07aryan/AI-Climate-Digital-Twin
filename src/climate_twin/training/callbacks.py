from dataclasses import dataclass
import numpy as np


@dataclass
class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    """

    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self):
        self.best_loss = np.inf
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Returns True if training should stop.
        """

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop

    def reset(self):
        self.best_loss = np.inf
        self.counter = 0
        self.stop = False