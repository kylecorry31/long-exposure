import cv2
import numpy as np

def apply_threshold(frame, threshold):
        converted_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, converted_frame = cv2.threshold(
            converted_frame, threshold, 255, cv2.THRESH_TOZERO
        )
        return converted_frame

def has_valid_pixels(frame, threshold, border_percentage = 0.0):
    if border_percentage > 0.0:
        h, w = frame.shape[:2]
        border_h = int(h * border_percentage)
        border_w = int(w * border_percentage)
        f = apply_threshold(frame[border_h:h-border_h, border_w:w-border_w], threshold)
    else:
        f = apply_threshold(frame, threshold)
    return not np.all(f == 0)