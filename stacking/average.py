from . import Stacker
import numpy as np

class AverageStacker(Stacker):
    def __init__(self):
        super().__init__()
        self._stack = None
        self._count = 0

    def stack(self, frame: np.ndarray):
        if self._stack is None:
            self._stack = frame
        else:
            self._stack = np.add(self._stack, frame)
        self._count += 1

    def get_image(self) -> np.ndarray:
        return self._stack / self._count