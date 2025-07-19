from . import Stacker
import numpy as np

class MedianStacker(Stacker):
    def __init__(self):
        super().__init__()
        self._frames = []

    def stack(self, frame: np.ndarray):
        self._frames.append(frame.astype(np.float64))

    def get_image(self) -> np.ndarray:
        if not self._frames:
            return None
        stacked = np.stack(self._frames, axis=0)
        return np.median(stacked, axis=0).astype(np.uint8)
