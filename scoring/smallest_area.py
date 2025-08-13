import numpy as np
from thresholding import apply_threshold
import cv2


class SmallestAreaScorer:
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def score(self, frame):
        moment = cv2.moments(apply_threshold(frame, self.threshold))
        m00 = moment["m00"]
        if m00 == 0:
            return 0.0
        return 1 / m00
