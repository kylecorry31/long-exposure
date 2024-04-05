import cv2
import os
import argparse
from reader.folder import FolderReader
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Create a video.")
parser.add_argument("input", help="path to image folder")
parser.add_argument("output", help="path to output file")
parser.add_argument("--fps", help="frames per second", type=int, default=30)

args = parser.parse_args()

reader = FolderReader(args.input)

video = None

with tqdm(total=reader.total_frames()) as pbar:
    while True:
        frame = reader.next_frame()
        if frame is None:
            break

        if video is None:
            height, width, _ = frame.shape
            # Record in avi format
            video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), args.fps, (width, height))
        
        video.write(frame)
        pbar.update(1)

video.release()




# def create_video(image_folder, output_name='output_video.mp4', fps=30):
#     images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape

#     video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     for image in images:
#         video.write(cv2.imread(os.path.join(image_folder, image)))

#     cv2.destroyAllWindows()
#     video.release()

# if __name__ == "__main__":
#     image_folder = input("Enter the directory containing images: ")
#     output_name = input("Enter the name of the output video (with .mp4 extension): ")
#     fps = int(input("Enter the frames per second (default is 30): ") or 30)
    
#     create_video(image_folder, output_name, fps)
#     print("Video created successfully!")
