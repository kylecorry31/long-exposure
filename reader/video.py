from . import Reader
import numpy as np
import cv2

class VideoReader(Reader):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._capture = None
    
    def next_frame(self) -> np.ndarray:
        if self._capture is None:
            self._capture = cv2.VideoCapture(self._path)
        ret, frame = self._capture.read()
        if not ret:
            return None
        return frame
    
    def total_frames(self) -> int:
        if self._capture is None:
            self._capture = cv2.VideoCapture(self._path)
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None