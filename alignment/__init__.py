import numpy as np

class Aligner(object):
    def __init__(self):
        self.reference = None

    def set_reference(self, reference: np.ndarray):
        self.reference = reference
    
    def align(self, frame: np.ndarray) -> np.ndarray:
        pass