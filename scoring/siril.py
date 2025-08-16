import numpy as np
import cv2


class SirilScorer:
    """
    Scorer implementation based on Siril's quality estimation algorithm.
    Vectorized for efficiency: block means, dilation-based mask, and gradient via array ops.
    """

    # Constants from Siril
    QSUBSAMPLE_MIN = 2
    QSUBSAMPLE_MAX = 16
    QSUBSAMPLE_INC = 2
    QMARGIN = 0.1
    THRESHOLD_8BIT = 30
    THRESHOLD_16BIT = 8000

    def __init__(self):
        super().__init__()

    def score(self, frame: np.ndarray) -> float:
        """
        Calculate quality score for the given frame.
        Args:
            frame: Input image (BGR or grayscale)
        Returns:
            Quality score (higher is better)
        """
        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Work in float32 for speed and to avoid overflow in gradients
        if gray.dtype == np.uint8:
            gray_f = gray.astype(np.float32)
            threshold = self.THRESHOLD_8BIT
            max_val = 255
        else:
            gray_f = gray.astype(np.float32)
            threshold = self.THRESHOLD_16BIT
            max_val = 65535

        return self._quality_estimate(gray_f, threshold, max_val)

    def _quality_estimate(self, image: np.ndarray, threshold: int, max_val: int) -> float:
        """
        Main quality estimation algorithm (vectorized).
        """
        height, width = image.shape

        # Region to analyze (full image minus borders, match original logic)
        region_w = width - 1
        region_h = height - 1

        dval = 0.0

        for subsample in range(self.QSUBSAMPLE_MIN, self.QSUBSAMPLE_MAX + 1, self.QSUBSAMPLE_INC):
            # Number of h & v pixels in subimage
            x_samples = region_w // subsample
            y_samples = region_h // subsample
            if x_samples < 2 or y_samples < 2:
                break

            # Trim to a region that's divisible by subsample (top-left)
            Ht = y_samples * subsample
            Wt = x_samples * subsample
            roi = image[0:Ht, 0:Wt]

            # Non-overlapping block mean via reshape (fast)
            subsampled = self._block_mean(roi, subsample)  # shape: (y_samples, x_samples), float32

            # Histogram stretching
            stretched = self._apply_histogram_stretch(subsampled, max_val)

            # 3x3 smoothing
            smoothed = self._smooth_image(stretched)

            # Gradient quality
            q = self._calculate_gradient(smoothed, threshold)

            if q > 0:
                dval += q * (self.QSUBSAMPLE_MIN * self.QSUBSAMPLE_MIN) / float(subsample * subsample)

        return float(np.sqrt(dval)) if dval > 0 else 0.0

    def _block_mean(self, img: np.ndarray, block: int) -> np.ndarray:
        """
        Compute non-overlapping block means using reshape.
        img shape must be (block*y, block*x)
        """
        h, w = img.shape
        y_blocks = h // block
        x_blocks = w // block
        # reshape to (y_blocks, block, x_blocks, block) and mean over the inner axes
        out = img.reshape(y_blocks, block, x_blocks, block).mean(axis=(1, 3))
        return out.astype(np.float32)

    def _apply_histogram_stretch(self, image: np.ndarray, max_val: int) -> np.ndarray:
        """
        Apply histogram stretching based on brightest pixels.
        Works on float32 arrays and returns float32.
        """
        # Use percentile-based stretching on positive pixels
        pos = image > 0
        if np.any(pos):
            p99 = np.percentile(image[pos], 99.0)
        else:
            p99 = 1.0

        if p99 <= 0:
            return image.astype(np.float32)

        target_max = 60000.0 if max_val > 255 else 240.0
        mult = target_max / float(p99)
        stretched = image * mult
        # Clip to 16-bit range to keep values bounded
        np.clip(stretched, 0.0, 65535.0, out=stretched)
        return stretched.astype(np.float32)

    def _smooth_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply 3x3 smoothing filter (average blur).
        """
        # cv2.blur is optimized; keep float32 to avoid overflow and keep speed
        smoothed = cv2.blur(image, ksize=(3, 3), borderType=cv2.BORDER_REPLICATE)
        return smoothed

    def _calculate_gradient(self, image: np.ndarray, threshold: int) -> float:
        """
        Vectorized gradient-based quality metric with 3x3 dilation mask.
        """
        h, w = image.shape
        yborder = int(h * self.QMARGIN) + 1
        xborder = int(w * self.QMARGIN) + 1

        if (yborder * 2 >= h - 2) or (xborder * 2 >= w - 2):
            return 0.0

        # Threshold mask then dilate by 3x3 to mark neighborhoods
        mask = (image >= float(threshold)).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)

        # Zero out margins
        mask[:yborder, :] = False
        mask[h - yborder :, :] = False
        mask[:, :xborder] = False
        mask[:, w - xborder :] = False

        # Compute forward differences (x and y), avoid last row/col for origins
        # Origins are pixels where we take differences to right and bottom
        origins = mask[:-1, :-1]
        if not origins.any():
            return 0.0

        dx = image[:-1, :-1] - image[:-1, 1:]
        dy = image[:-1, :-1] - image[1:, :-1]
        g2 = dx * dx + dy * dy

        val = g2[origins].mean()
        val /= 10.0  # scale factor from original algorithm
        return float(val)
