import numpy as np
import cv2

class SharpnessScorer:
    def __init__(self):
        super().__init__()

    def score(self, frame):
        # Calculate sharpness of the image
        # Convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate the Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Calculate the variance
        return np.var(laplacian)