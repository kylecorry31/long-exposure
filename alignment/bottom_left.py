from . import Aligner
import cv2
import numpy as np


class BottomLeftAligner(Aligner):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold

    def apply_threshold(self, frame):
        converted_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        _, converted_frame = cv2.threshold(
            converted_frame, self.threshold, 255, cv2.THRESH_TOZERO
        )
        return converted_frame

    def set_reference(self, reference):
        self.reference = self.apply_threshold(reference)

    def align(self, frame):
        if self.reference is None:
            self.set_reference(frame)
            self.reference_contours, _ = cv2.findContours(self.reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.reference_rect = cv2.boundingRect(max(self.reference_contours, key=cv2.contourArea))
            return frame

        # Convert the current frame to grayscale
        gray_frame = self.apply_threshold(frame)

        # Get the min rectangle that bounds the reference and current frame
        frame_contours, _ = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_rect = cv2.boundingRect(max(frame_contours, key=cv2.contourArea))

        # Calculate the translation vector to align the bottom left corner of the current frame to the reference frame
        translation_x = self.reference_rect[0] - frame_rect[0]
        translation_y = self.reference_rect[1] - frame_rect[1]

        rows, cols = self.reference.shape

        # Warp the current frame (translation)
        M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        aligned_frame = cv2.warpAffine(frame, M, (cols, rows))

        return aligned_frame
