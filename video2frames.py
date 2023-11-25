import cv2
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='path to the video to be converted to frames')
    parser.add_argument('--percent', type=int, default=10, help='save a frame at every this percent of the video')
    args = parser.parse_args()

    assert args.path is not None, "Please provide the path to the video to be converted to frames"

    # save dir
    save_dir = "frames"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # video
    video = cv2.VideoCapture(args.path)

    # fps of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"FPS of the video: {fps}")

    # Total frames in the video
    frames_tot = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {frames_tot}")

    # every percent'th frame
    frame_every_percent = int(frames_tot * args.percent / 100)

    # convert to & save frames
    success = True
    count = 0
    i = 0
    while success:
        success, image = video.read()
        if count % frame_every_percent == 0:
            cv2.imwrite(os.path.join(save_dir, "frame%d.jpg" %i), image) # save frame as JPEG file
            print(f"A frame captured at {i * args.percent}th % of the video.")
            i += 1
        count += 1
