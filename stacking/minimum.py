from . import Stacker
import numpy as np

class MinimumStacker(Stacker):
    def __init__(self):
        super().__init__()
        self._stack = None

    def stack(self, frame: np.ndarray):
        if self._stack is None:
            self._stack = frame
        else:
            self._stack = np.minimum(self._stack, frame)

    def get_image(self) -> np.ndarray:
        return self._stack