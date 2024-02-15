from . import Aligner
import cv2
import numpy as np
from scipy.ndimage import shift

class FFTAligner(Aligner):
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

        # Compute the 2d FFTs
        f1 = np.fft.fft2(self.reference)
        f2 = np.fft.fft2(gray_frame)

        # Compute the cross power spectrum
        cross_power_spectrum = f1 * np.conj(f2)
        cross_power_spectrum /= np.abs(cross_power_spectrum)

        # Compute the inverse FFT to obtain the translation
        translation = np.fft.ifft2(cross_power_spectrum).real

        # Find the peak of the translation to get the shift
        shift_y, shift_x = np.unravel_index(np.argmax(translation), translation.shape)

        # Shift the original frame to align it with the reference
        aligned_frame = shift(frame, [shift_y, shift_x, 0])

        return aligned_frame