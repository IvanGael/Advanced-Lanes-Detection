"""
@Author Ivan APEDO

Advanced Lane Lines Detection

Usage:
    py main.py --choice CHOICE --input INPUT_PATH --output OUTPUT_PATH 
"""

import argparse
import os
import numpy as np
import matplotlib.image as mpimg
import cv2
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from yolo import CarDetection

class FindLaneLines:
    def __init__(self):
        self.calibration = CameraCalibration(chessboard_size=(9, 6))
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()
        self.car_detection = CarDetection()
        self.camera_matrix = None  # To store camera matrix after calibration

    def calibrate_camera_for_video(self, video_path):
        # Extract frames from video for calibration
        frames = self.calibration.extract_frames(video_path)
        # Perform calibration
        calibrated = self.calibration.calibrate(frames)
        if calibrated:
            print("Camera calibration successful!")
        else:
            print("Camera calibration failed. Make sure the frames contain the chessboard pattern.")

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)

        # Perform car detection
        results = self.car_detection.detect_cars(img)

        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)

        annotated_img = self.car_detection.draw_boxes(results)

        # Generate aerial (top) view
        top_view_img = self.transform.get_aerial_view(out_img)

        # Overlay aerial view at top center
        overlay_img = self.transform.overlay_aerial_view(out_img, top_view_img)

        # Combine the final images
        final_img = cv2.addWeighted(overlay_img, 0.7, annotated_img, 0.3, 0)

        return final_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        # Perform dynamic camera calibration for this video
        self.calibrate_camera_for_video(input_path)

        # Process the video
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)


def main():
    parser = argparse.ArgumentParser(description="Advanced Lane Lines Detection")
    parser.add_argument("--choice", choices=['video', 'image'], default='video',
                        help="Choose between 'video' and 'image' (default: video)")
    parser.add_argument("--input", required=True, help="Choose an input video or image")
    parser.add_argument("--output", help="Choose an output video or image")
    args = parser.parse_args()

    input_file = f"output_videos/{args.input}"
    output_file = args.output

    # if output_file is not provided
    if output_file is None:
        input_name, input_ext = os.path.splitext(os.path.basename(input_file))
        if args.choice == 'video':
            output_file = f"output_videos/{input_name}_output.mp4"
        else:
            output_file = f"output_videos/{input_name}_output{input_ext}"

    findLaneLines = FindLaneLines()

    if args.choice == 'video':
        findLaneLines.process_video(input_file, output_file)
    else:
        findLaneLines.process_image(input_file, output_file)

if __name__ == "__main__":
    main()
