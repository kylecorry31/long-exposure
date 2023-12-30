from alignment.null import NullAligner
from alignment.sift import SiftAligner
from alignment.ecc import EccAligner
from stacking.maximum import MaximumStacker
from stacking.average import AverageStacker
from stacking.minimum import MinimumStacker
from reader.video import VideoReader
from reader.folder import FolderReader
import argparse
from tqdm import tqdm
import cv2
import os

# Parse arguments
parser = argparse.ArgumentParser(description='Stack images from a video.')
parser.add_argument('input', help='path to video file or image folder')
parser.add_argument('output', help='path to output file')
parser.add_argument('--align', help='alignment method', choices=['sift', 'ecc', 'none'], default='none')
parser.add_argument('--stack', help='stacking method', choices=['max', 'avg', 'min'], default='avg')
parser.add_argument('--threshold', help='brightness threshold for alignment', type=float, default=0.0)

args = parser.parse_args()

# Read video
if os.path.isdir(args.input):
    reader = FolderReader(args.input)
else:
    reader = VideoReader(args.input)

# Create aligner
if args.align == 'sift':
    aligner = SiftAligner(args.threshold)
elif args.align == 'ecc':
    aligner = EccAligner(args.threshold)
else:
    aligner = NullAligner()

# Create stacker
if args.stack == 'max':
    stacker = MaximumStacker()
elif args.stack == 'min':
    stacker = MinimumStacker()
else:
    stacker = AverageStacker()

# Stack frames
with tqdm(total=reader.total_frames()) as pbar:
    while True:
        frame = reader.next_frame()
        if frame is None:
            break
        frame = aligner.align(frame)
        stacker.stack(frame)
        pbar.update(1)

# Save stacked image
stacked = stacker.get_image()

output_path = args.output

cv2.imwrite(output_path, stacked)

# Close reader
reader.close()