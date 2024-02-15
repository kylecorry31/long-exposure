from . import Aligner
import cv2
import numpy as np


class OrbAligner(Aligner):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.orb = cv2.ORB_create(nfeatures=5000)  # Increase the number of features
        index_params = dict(
            algorithm=6, table_number=6, key_size=12, multi_probe_level=1
        )
        search_params = dict(checks=100)  # Increase the number of checks
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.threshold = threshold

    def apply_threshold(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)  # Apply histogram equalization
        _, frame_bin = cv2.threshold(frame_gray, self.threshold, 255, cv2.THRESH_BINARY)
        return frame_bin

    def align(self, frame):
        if self.reference is None:
            self.set_reference(frame)
            return frame

        frame_gray = self.apply_threshold(frame)
        kp2, des2 = self.orb.detectAndCompute(frame_gray, None)

        matches = self.matcher.knnMatch(self.ref_descriptors, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:  # Adjust this ratio as needed
                good_matches.append(m)

        points1 = np.float32(
            [self.ref_keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        height, width = self.reference.shape
        im2_aligned = cv2.warpPerspective(frame, h, (width, height))

        return im2_aligned
