import numpy as np

class Reader(object):
    def __init__(self):
        pass
    
    def next_frame(self) -> np.ndarray:
        return None
    
    def skip_next_frame(self):
        pass
    
    def total_frames(self) -> int:
        return 0
    
    def close(self):
        pass