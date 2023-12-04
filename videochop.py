import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None, help='path to the video to be converted to frames')
parser.add_argument('--save_dir', type=str, default=None, help='directory to save the frames')
args = parser.parse_args()

assert args.path is not None, "Please provide the path to the video to be converted to frames"

save_dir = args.save_dir
vidcap = cv2.VideoCapture(args.path)
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite(os.path.join(save_dir, "frame%d.jpg" % count), image)
    success,image = vidcap.read()
    count += 1
