from . import Aligner
import cv2
import numpy as np


class SiftAligner(Aligner):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.threshold = threshold

    def apply_threshold(self, frame):
        converted_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, converted_frame = cv2.threshold(converted_frame, self.threshold, 255, cv2.THRESH_TOZERO)
        return converted_frame

    def set_reference(self, reference):
        self.reference = self.apply_threshold(reference)
        self.ref_keypoints, self.ref_descriptors = self.sift.detectAndCompute(self.reference, None)
        if self.ref_descriptors.dtype != np.float32:
            self.ref_descriptors = self.ref_descriptors.astype(np.float32)

    def align(self, frame):
        if self.reference is None:
            self.set_reference(frame)
            return frame

        # Convert images to grayscale
        frame_gray = self.apply_threshold(frame)

        # Find keypoints and descriptors for the second image
        kp2, des2 = self.sift.detectAndCompute(frame_gray, None)

        if des2.dtype != np.float32:
            des2 = des2.astype(np.float32)

        matches = self.matcher.knnMatch(self.ref_descriptors, des2, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract location of good matches
        points1 = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Compute homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use homography to align images
        height, width = self.reference.shape
        im2_aligned = cv2.warpPerspective(frame, h, (width, height))

        return im2_aligned
