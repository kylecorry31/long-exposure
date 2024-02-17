from scoring import Scorer
from . import Reader
import numpy as np
from tqdm import tqdm

class SortedReader(Reader):

    def __init__(self, reader: Reader, scorer: Scorer, keepPercentage: float = 1.0):
        super().__init__()
        self.frames = []
        scores = []
        self.index = 0

        with tqdm(total=reader.total_frames(), desc="Scoring frames") as pbar:
            for i in range(reader.total_frames()):
                frame = reader.next_frame()
                if frame is None:
                    break
                self.frames.append(frame)
                scores.append(scorer.score(frame))
                pbar.update(1)
        reader.close()

        # Sort the frames by score
        sortedIndices = sorted(range(len(scores)), key=lambda k: scores[k])
        self.frames = [self.frames[i] for i in sortedIndices]
        # Remove the bottom keepPercentage of frames
        self.frames = list(reversed(self.frames[int(len(self.frames) * (1 - keepPercentage)):]))

    
    def next_frame(self) -> np.ndarray:
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return frame
    
    def total_frames(self) -> int:
        return len(self.frames)
    
    def close(self):
        self.frames = []