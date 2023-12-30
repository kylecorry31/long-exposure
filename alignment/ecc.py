from . import Aligner
import cv2
import numpy as np


class EccAligner(Aligner):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold

    def apply_threshold(self, frame):
        converted_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, converted_frame = cv2.threshold(converted_frame, self.threshold, 255, cv2.THRESH_TOZERO)
        return converted_frame

    def set_reference(self, reference):
        self.reference = self.apply_threshold(reference)
    
    def align(self, frame):
        if self.reference is None:
            self.set_reference(frame)
            return frame

        # Convert the current frame to grayscale
        gray_frame = self.apply_threshold(frame)
        # Estimate the geometric transformation that aligns the current frame to the reference frame
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        _, warp_matrix = cv2.findTransformECC(self.reference, gray_frame, warp_matrix, cv2.MOTION_AFFINE, criteria)
        # Apply the transformation to the frame
        rows, cols = frame.shape[:2]
        return cv2.warpAffine(frame, warp_matrix, (cols, rows))
