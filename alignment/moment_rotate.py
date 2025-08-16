from . import Aligner
import cv2
import numpy as np
from thresholding import apply_threshold


class MomentRotateAligner(Aligner):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold

    def apply_threshold(self, frame):
        return apply_threshold(frame, self.threshold, True)

    def set_reference(self, reference):
        self.reference = self.apply_threshold(reference)
        self.reference_moment = cv2.moments(self.reference, True)
        self.reference_hu_moment = cv2.HuMoments(self.reference_moment)

    def align(self, frame):
        if self.reference is None:
            self.set_reference(frame)
            return frame

        # Convert the current frame to grayscale
        gray_frame = self.apply_threshold(frame)

        # Find the image moment of the reference and current frame
        frame_moment = cv2.moments(gray_frame, True)

        # Calculate the centroid of the reference and current frame
        m00 = frame_moment["m00"]
        if m00 == 0:
            print("Warning: Frame moment is zero, cannot align frame.")
            return None

        reference_centroid_x = int(self.reference_moment["m10"] / self.reference_moment["m00"])
        reference_centroid_y = int(self.reference_moment["m01"] / self.reference_moment["m00"])
        frame_centroid_x = int(frame_moment["m10"] / m00)
        frame_centroid_y = int(frame_moment["m01"] / m00)

        # Calculate the translation vector
        translation_x = reference_centroid_x - frame_centroid_x
        translation_y = reference_centroid_y - frame_centroid_y

        # Calculate the scale based on the ratio of reference moment to frame moment
        scale = np.sqrt(self.reference_moment["m00"] / m00)

        # Calculate the angle of rotation based on the first Hu moment
        frame_hu_moment = cv2.HuMoments(frame_moment)
        angle = np.arctan2(
            self.reference_hu_moment[0] - frame_hu_moment[0],
            frame_hu_moment[0] + self.reference_hu_moment[0],
        )[0]

        rows, cols = self.reference.shape

        # Scale and rotate the current frame with the frame centroid as the pivot
        M = cv2.getRotationMatrix2D(
            (frame_centroid_x, frame_centroid_y), np.degrees(angle), scale
        )
        aligned_frame = cv2.warpAffine(frame, M, (cols, rows))

        # Warp the current frame (translation)
        M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        aligned_frame = cv2.warpAffine(aligned_frame, M, (cols, rows))

        return aligned_frame
