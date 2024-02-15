import numpy as np

class ContrastScorer:
    def __init__(self):
        super().__init__()

    def score(self, frame):
        # Calculate contrast of the image
        mean = np.mean(frame)
        std = np.std(frame)
        return std / mean