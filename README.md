## Advanced Lane Detection

An advanced lane detection system using Using OpenCV, canny edge detector and hough transform algorithms

![Demo](demo.png)

The Project
---

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position to the center.
* Warp the detected lane boundaries back onto the original image.
* Lane Width: Computes the width of the lane based on polynomial coefficients.
* Get Top Down View of the lane using "birds-eye view" technique
* Vehicles Detection with YOLO11 by Ultralytics
* Distance Estimation of each Vehicle from others based on the centroids of the bounding boxes

The images for camera calibration are stored in the folder called `camera_cal`.


### Requirements
```bash
pip install -r requirements.txt
```

### Run 
```bash
py main.py --choice CHOICE --input INPUT_PATH 
```
