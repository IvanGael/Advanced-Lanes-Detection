"""
@Author Ivan APEDO

Advanced Lane Lines Detection

Usage:
    py main.py --choice CHOICE --input INPUT_PATH --output OUTPUT_PATH 
"""

import argparse
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
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()
        self.car_detection = CarDetection()  

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        
        # Perform car detection
        car_detection_img = self.car_detection.process_image(img)
        
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        
        # Combine lane detection and car detection
        final_img = cv2.addWeighted(out_img, 0.7, car_detection_img, 0.3, 0)
        
        return final_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def main():
    parser = argparse.ArgumentParser(description="Advanced Lane Lines Detection")
    parser.add_argument("--choice", choices=['video', 'image'], default='video',
                        help="Choose between 'video' and 'image' (default: video)")
    parser.add_argument("--input", help="Choose an input video or image")
    parser.add_argument("--output", help="Choose an output video or image")


    args = parser.parse_args()

    input = args.input
    output = args.output

    findLaneLines = FindLaneLines()

    if args.choice == 'video':
        findLaneLines.process_video(input, output)
    else:
        findLaneLines.process_image(input, output)

if __name__ == "__main__":
    main()