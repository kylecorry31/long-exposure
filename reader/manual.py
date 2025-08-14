from . import Reader
import numpy as np
import cv2

class ManualReader(Reader):
    """
    Wraps another Reader and lets the user manually pick which frames to keep.
    An interactive window is shown with three buttons: Keep, Discard, and Discard Remaining.
    - Keep stores the current frame index.
    - Discard skips the current frame.
    - Discard Remaining stops the review early and discards all remaining frames.
    After selection, the wrapper behaves like a reader over the kept frames, similar to SortedReader.
    """

    def __init__(self, reader: Reader, scale: float = 1.0):
        super().__init__()
        self.reader = reader
        self.frames = []  # indices of kept frames from the underlying reader
        self.index = 0
        self._window_name = "Manual"
        self._button_regions = {}  # name -> (x1, y1, x2, y2)
        self._selection = None  # one of 'keep', 'discard', 'discard_remaining'
        self.scale = scale

        total = reader.total_frames()
        if total <= 0:
            return

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._window_name, self._on_mouse)

        for i in range(total):
            frame = reader.next_frame()
            if frame is None:
                break

            if self.scale > 0:
                frame = cv2.convertScaleAbs(frame, alpha=self.scale, beta=0)

            # Render UI frame with buttons and index overlay
            display = self._render_ui(frame, i, total)

            # Reset selection for this frame
            self._selection = None

            # Show until a selection is made via mouse or keyboard
            while True:
                cv2.imshow(self._window_name, display)
                key = cv2.waitKey(20) & 0xFF
                if key == ord('k'):
                    self._selection = 'keep'
                elif key == ord('d'):
                    self._selection = 'discard'
                elif key == ord('q'):
                    self._selection = 'discard_remaining'

                if self._selection is not None:
                    break

            if self._selection == 'keep':
                self.frames.append(i)
            elif self._selection == 'discard_remaining':
                break
            # if discard, do nothing and continue

        # Cleanup UI and reset underlying reader so we can access kept frames later
        cv2.destroyWindow(self._window_name)
        self.reader.reset()

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            for name, (x1, y1, x2, y2) in self._button_regions.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._selection = name
                    break

    def _render_ui(self, frame: np.ndarray, index: int, total: int) -> np.ndarray:
        h, w = frame.shape[:2]
        footer = 70
        canvas = np.zeros((h + footer, w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame

        # Footer background
        canvas[h:, :] = (30, 30, 30)

        # Button layout
        pad = 10
        btn_top = h + 10
        btn_bottom = h + 50
        btn_width = max(120, (w - pad * 4) // 3)

        # Keep button (green)
        kx1 = pad
        kx2 = min(w - pad, kx1 + btn_width)
        cv2.rectangle(canvas, (kx1, btn_top), (kx2, btn_bottom), (0, 180, 0), -1)
        cv2.putText(canvas, "Keep (k)", (kx1 + 12, btn_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Discard button (red)
        dx1 = kx2 + pad
        dx2 = min(w - pad, dx1 + btn_width)
        cv2.rectangle(canvas, (dx1, btn_top), (dx2, btn_bottom), (0, 0, 180), -1)
        cv2.putText(canvas, "Discard (d)", (dx1 + 12, btn_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Discard remaining button (gray)
        rx1 = dx2 + pad
        rx2 = min(w - pad, rx1 + btn_width)
        cv2.rectangle(canvas, (rx1, btn_top), (rx2, btn_bottom), (80, 80, 80), -1)
        cv2.putText(canvas, "Discard Remaining (q)", (rx1 + 12, btn_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Save button regions for click handling
        self._button_regions = {
            'keep': (kx1, btn_top, kx2, btn_bottom),
            'discard': (dx1, btn_top, dx2, btn_bottom),
            'discard_remaining': (rx1, btn_top, rx2, btn_bottom),
        }

        # Index overlay (top-left)
        cv2.rectangle(canvas, (5, 5), (160, 35), (0, 0, 0), -1)
        cv2.putText(canvas, f"{index + 1}/{total}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    # Reader interface using kept indices
    def next_frame(self) -> np.ndarray:
        if self.index >= len(self.frames):
            return None
        frame = self.reader.get_frame(self.frames[self.index])
        self.index += 1
        return frame

    def get_frame(self, i) -> np.ndarray:
        if i < 0 or i >= len(self.frames):
            return None
        return self.reader.get_frame(self.frames[i])

    def reset(self):
        self.index = 0

    def skip_next_frame(self):
        self.index += 1

    def total_frames(self) -> int:
        return len(self.frames)

    def close(self):
        self.frames = []
