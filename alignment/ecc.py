from . import Aligner
import cv2
import numpy as np


class EccAligner(Aligner):
    def __init__(self):
        super().__init__()

    def set_reference(self, reference):
        self.reference = cv2.cvtColor(reference.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    def align(self, frame):
        if self.reference is None:
            self.set_reference(frame)
            return frame

        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # Estimate the geometric transformation that aligns the current frame to the reference frame
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        _, warp_matrix = cv2.findTransformECC(self.reference, gray_frame, warp_matrix, cv2.MOTION_AFFINE, criteria)
        # Apply the transformation to the frame
        rows, cols = frame.shape[:2]
        return cv2.warpAffine(frame, warp_matrix, (cols, rows))
