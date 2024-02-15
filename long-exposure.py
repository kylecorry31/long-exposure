from alignment.fft import FFTAligner
from alignment.fourier_shift import FourierShiftAligner
from alignment.moment_rotate import MomentRotateAligner
from alignment.null import NullAligner
from alignment.orb import OrbAligner
from alignment.sift import SiftAligner
from alignment.ecc import EccAligner
from alignment.moment import MomentAligner
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
parser = argparse.ArgumentParser(description="Stack images from a video.")
parser.add_argument("input", help="path to video file or image folder")
parser.add_argument("output", help="path to output file")
parser.add_argument(
    "--align",
    help="alignment method",
    choices=["sift", "ecc", "moment", "moment_rotate", "orb", "fourier", "fft", "none"],
    default="none",
)
parser.add_argument(
    "--stack", help="stacking method", choices=["max", "avg", "min"], default="avg"
)
parser.add_argument(
    "--threshold", help="brightness threshold for alignment", type=float, default=0.0
)

args = parser.parse_args()

# Read video
if os.path.isdir(args.input):
    reader = FolderReader(args.input)
else:
    reader = VideoReader(args.input)

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
else:
    aligner = NullAligner()

# Create stacker
if args.stack == "max":
    stacker = MaximumStacker()
elif args.stack == "min":
    stacker = MinimumStacker()
else:
    stacker = AverageStacker()

aligned_folder = args.input + "_aligned"
if not os.path.exists(aligned_folder):
    os.makedirs(aligned_folder)

# Stack frames
i = 0
with tqdm(total=reader.total_frames()) as pbar:
    while True:
        frame = reader.next_frame()
        i += 1
        if frame is None:
            break
        frame = aligner.align(frame)
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
