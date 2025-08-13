from . import Reader
import numpy as np
import cv2
import os

class FolderReader(Reader):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._index = 0
        self._files = os.listdir(path)
        self._files.sort()

        # Sort by the number in the filename
        # self._files.sort(key=lambda x: int(x.split('.')[0]))

        # Filter out non-image files
        self._files = [f for f in self._files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self._count = len(self._files)
    
    def next_frame(self) -> np.ndarray:
        if self._index >= self._count:
            return None
        frame = cv2.imread(os.path.join(self._path, self._files[self._index]))
        self._index += 1
        return frame
    
    def get_frame(self, i) -> np.ndarray:
        if i < 0 or i >= self._count:
            return None
        return cv2.imread(os.path.join(self._path, self._files[i]))

    def reset(self):
        self._index = 0

    def skip_next_frame(self):
        self._index += 1
    
    def total_frames(self) -> int:
        return self._count
    
    def close(self):
        pass