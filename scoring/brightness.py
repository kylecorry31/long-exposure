import numpy as np


class BrightnessScorer:
    def __init__(self):
        super().__init__()

    def score(self, frame):
        return np.mean(frame)
