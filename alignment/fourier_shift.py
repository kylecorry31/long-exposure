import numpy as np
from scipy.ndimage import fourier_shift
import cv2
from alignment import Aligner


class FourierShiftAligner(Aligner):
    def __init__(self):
        super().__init__()
        self.reference = None

    def set_reference(self, frame):
        self.reference = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def align(self, frame):
        if self.reference is None:
            self.set_reference(frame)
            return frame

        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute cross-power spectrum
        cross_power_spectrum = np.fft.fft2(self.reference) * np.fft.fft2(frame).conj()
        cross_power_spectrum /= np.abs(cross_power_spectrum)

        # Compute cross-correlation
        cross_correlation = np.fft.ifft2(cross_power_spectrum).real

        # Find peak in cross-correlation
        shifts = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)

        # Shift the image
        shifted_image = fourier_shift(np.fft.fft2(frame), shifts)
        shifted_image = np.fft.ifft2(shifted_image)

        return np.abs(shifted_image)