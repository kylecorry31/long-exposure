from . import Aligner

class NullAligner(Aligner):
    def __init__(self):
        super().__init__()

    def set_reference(self, reference):
        self.reference = reference

    def align(self, frame):
        return frame