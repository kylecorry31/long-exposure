from alignment.bottom_left import BottomLeftAligner
from alignment.fft import FFTAligner
from alignment.fourier_shift import FourierShiftAligner
from alignment.moment_rotate import MomentRotateAligner
from alignment.null import NullAligner
from alignment.orb import OrbAligner
from alignment.sift import SiftAligner
from alignment.ecc import EccAligner
from alignment.moment import MomentAligner
from scoring.contrast import ContrastScorer
from scoring.sharpness import SharpnessScorer
from stacking.maximum import MaximumStacker
from stacking.average import AverageStacker
from stacking.median import MedianStacker
from stacking.minimum import MinimumStacker
from reader.video import VideoReader
from reader.folder import FolderReader
from reader.sorted import SortedReader
from scoring.brightness import BrightnessScorer
import argparse
from tqdm import tqdm
import cv2
import os

# Parse arguments
parser = argparse.ArgumentParser(description="Stack images from a video.")
parser.add_argument("input", help="path to video file or image folder")
parser.add_argument("output", help="path to output file")
parser.add_argument(
    "--align",
    help="alignment method",
    choices=["sift", "ecc", "moment", "moment_rotate", "orb", "fourier", "fft", "bottom-left", "none"],
    default="none",
)
parser.add_argument(
    "--stack", help="stacking method", choices=["max", "avg", "min", "median"], default="avg"
)
parser.add_argument(
    "--top", help="percent of top frames to stack", type=float, default=100.0
)
parser.add_argument(
    "--threshold", help="brightness threshold for alignment", type=float, default=0.0
)
parser.add_argument(
    "--score",
    help="scoring method",
    choices=["contrast", "brightness", "sharpness", "none"],
    default="none",
)
parser.add_argument(
    "--rotation",
    help="rotation angle of the image",
    type=int,
    default=0
)
parser.add_argument(
    "--step",
    help="step to skip frames",
    type=int,
    default=1
)
parser.add_argument(
    "--mask",
    help="whether to use the threshold as a mask",
    type=bool,
    default=False
)

args = parser.parse_args()

# Read video
if os.path.isdir(args.input):
    reader = FolderReader(args.input)
else:
    reader = VideoReader(args.input)

if args.score == "brightness":
    reader = SortedReader(reader, BrightnessScorer(), args.top / 100.0)
elif args.score == "contrast":
    reader = SortedReader(reader, ContrastScorer(), args.top / 100.0)
elif args.score == "sharpness":
    reader = SortedReader(reader, SharpnessScorer(), args.top / 100.0)
elif args.top < 100.0:
    reader = SortedReader(reader, ContrastScorer(), args.top / 100.0)

# Create aligner
if args.align == "sift":
    aligner = SiftAligner(args.threshold)
elif args.align == "ecc":
    aligner = EccAligner(args.threshold)
elif args.align == "moment":
    aligner = MomentAligner(args.threshold)
elif args.align == "orb":
    aligner = OrbAligner(args.threshold)
elif args.align == "moment_rotate":
    aligner = MomentRotateAligner(args.threshold)
elif args.align == "fourier" or args.align == "fft":
    aligner = FFTAligner(args.threshold)
elif args.align == "bottom-left":
    aligner = BottomLeftAligner(args.threshold)
else:
    aligner = NullAligner()

# Create stacker
if args.stack == "max":
    stacker = MaximumStacker()
elif args.stack == "min":
    stacker = MinimumStacker()
elif args.stack == "median":
    stacker = MedianStacker()
else:
    stacker = AverageStacker()

aligned_folder = args.input + "_aligned"
if not os.path.exists(aligned_folder):
    os.makedirs(aligned_folder)

# Stack frames
i = 0
with tqdm(total=reader.total_frames()) as pbar:
    while True:
        if i % args.step != 0:
            pbar.update(1)
            reader.skip_next_frame()
            i += 1
            continue

        i += 1
        frame = reader.next_frame()

        if frame is None:
            break
        # Rotate the frame
        if args.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        frame = aligner.align(frame)

        # Apply mask
        if args.mask:
            mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args.threshold, 255)
            frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Save frame back to a subfolder
        frame_path = os.path.join(aligned_folder, f"{i}.jpg")
        cv2.imwrite(frame_path, frame)

        stacker.stack(frame)
        pbar.update(1)

# Save stacked image
stacked = stacker.get_image()

output_path = args.output

cv2.imwrite(output_path, stacked)

# Close reader
reader.close()
