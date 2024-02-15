from . import Aligner
import cv2
import numpy as np


class MomentRotateAligner(Aligner):
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
            return frame

        # Convert the current frame to grayscale
        gray_frame = self.apply_threshold(frame)

        # Find the image moment of the reference and current frame
        reference_moment = cv2.moments(self.reference)
        frame_moment = cv2.moments(gray_frame)

        # Calculate the centroid of the reference and current frame
        reference_centroid_x = int(reference_moment["m10"] / reference_moment["m00"])
        reference_centroid_y = int(reference_moment["m01"] / reference_moment["m00"])
        frame_centroid_x = int(frame_moment["m10"] / frame_moment["m00"])
        frame_centroid_y = int(frame_moment["m01"] / frame_moment["m00"])

        # Calculate the translation vector
        translation_x = reference_centroid_x - frame_centroid_x
        translation_y = reference_centroid_y - frame_centroid_y

        # Calculate the scale based on the ratio of reference moment to frame moment
        scale = np.sqrt(reference_moment['m00'] / frame_moment['m00'])

        # Calculate the angle of rotation based on the first Hu moment
        reference_hu_moment = cv2.HuMoments(reference_moment)
        frame_hu_moment = cv2.HuMoments(frame_moment)
        angle = np.arctan2(reference_hu_moment[0] - frame_hu_moment[0], frame_hu_moment[0] + reference_hu_moment[0])[0]

        rows, cols = self.reference.shape

        # Scale and rotate the current frame with the frame centroid as the pivot
        M = cv2.getRotationMatrix2D((frame_centroid_x, frame_centroid_y), np.degrees(angle), scale)
        aligned_frame = cv2.warpAffine(frame, M, (cols, rows))

        # Warp the current frame (translation)
        M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        aligned_frame = cv2.warpAffine(aligned_frame, M, (cols, rows))

        return aligned_frame
