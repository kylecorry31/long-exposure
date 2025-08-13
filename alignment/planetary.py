from . import Aligner
import cv2
import numpy as np


class PlanetaryAligner(Aligner):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = float(threshold)
        self.reference_high = None
        self.reference_low = None

    def apply_threshold(self, frame, factor: float = 1.0):
        converted_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        thr = max(0.0, self.threshold * float(factor))
        _, converted_frame = cv2.threshold(converted_frame, thr, 255, cv2.THRESH_TOZERO)
        return converted_frame

    def set_reference(self, reference):
        self.reference_high = self.apply_threshold(reference, 1.3)
        self.reference_low = self.apply_threshold(reference, 0.7)
        self.reference = self.reference_high

    def _centroid_and_scale(self, ref_gray: np.ndarray, frame_gray: np.ndarray):
        ref_m = cv2.moments(ref_gray)
        frm_m = cv2.moments(frame_gray)
        m00_ref = ref_m.get("m00", 0.0)
        m00_frm = frm_m.get("m00", 0.0)
        if m00_ref == 0 or m00_frm == 0:
            return None
        ref_cx = int(ref_m["m10"] / m00_ref)
        ref_cy = int(ref_m["m01"] / m00_ref)
        frm_cx = int(frm_m["m10"] / m00_frm)
        frm_cy = int(frm_m["m01"] / m00_frm)
        tx = ref_cx - frm_cx
        ty = ref_cy - frm_cy
        scale = float(np.sqrt(m00_ref / m00_frm))
        return (ref_cx, ref_cy, frm_cx, frm_cy, tx, ty, scale)

    def _orientation_angle_deg(self, gray: np.ndarray):
        m = cv2.moments(gray)
        if m.get("m00", 0.0) == 0:
            return 0.0
        denom = (m["mu20"] - m["mu02"]) if (m["mu20"] - m["mu02"]) != 0 else 1e-12
        angle_rad = 0.5 * np.arctan2(2.0 * m["mu11"], denom)
        return float(np.degrees(angle_rad))

    def _difference_metric(self, a_gray: np.ndarray, b_gray: np.ndarray) -> float:
        a32 = a_gray.astype(np.float32)
        b32 = b_gray.astype(np.float32)
        return float(np.mean(np.abs(a32 - b32)))

    def align(self, frame):
        if self.reference_high is None or self.reference_low is None:
            self.set_reference(frame)
            return frame

        rows, cols = self.reference_high.shape

        # Step 1: Translation/scaling
        frame_high = self.apply_threshold(frame, 1.3)
        params = self._centroid_and_scale(self.reference_high, frame_high)
        if params is None:
            return None
        ref_cx, ref_cy, frm_cx, frm_cy, tx, ty, scale = params

        M_scale = cv2.getRotationMatrix2D((frm_cx, frm_cy), 0, scale)
        new_frame_high = cv2.warpAffine(frame, M_scale, (cols, rows))
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        new_frame_high = cv2.warpAffine(new_frame_high, M_trans, (cols, rows))

        # Step 2: Rotation correction
        ref_low = self.reference_low
        new_frame_low = self.apply_threshold(new_frame_high, 0.7)

        reference_angle = self._orientation_angle_deg(ref_low)
        frame_angle = self._orientation_angle_deg(new_frame_low)
        base_delta = reference_angle - frame_angle

        best_frame = new_frame_high
        best_diff = float("inf")
        best_angle = base_delta

        m_low = cv2.moments(new_frame_low)
        if m_low.get("m00", 0.0) == 0:
            return new_frame_high
        cx = int(m_low["m10"] / m_low["m00"]) if m_low["m00"] != 0 else cols // 2
        cy = int(m_low["m01"] / m_low["m00"]) if m_low["m00"] != 0 else rows // 2

        def eval_angle(angle_deg: float):
            M_rot = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
            candidate = cv2.warpAffine(
                new_frame_high,
                M_rot,
                (cols, rows),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            cand_low = self.apply_threshold(candidate, 0.9)
            diff = self._difference_metric(ref_low, cand_low)
            return diff, candidate

        # High level alignment
        for off in np.arange(-10.0, 10.0001, 0.5):
            angle = base_delta + float(off)
            diff, cand = eval_angle(angle)
            if diff < best_diff:
                best_diff = diff
                best_frame = cand
                best_angle = angle

        # Low level alignment
        for off in np.arange(-1.0, 1.0001, 0.1):
            angle = best_angle + float(off)
            diff, cand = eval_angle(angle)
            if diff < best_diff:
                best_diff = diff
                best_frame = cand
                best_angle = angle

        return best_frame
