import cv2
import numpy as np
import os

class CameraCalibration:
    def __init__(self, chessboard_size=(9, 6)):
        self.chessboard_size = chessboard_size
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        self.mtx = None
        self.dist = None

    def extract_frames(self, video_path, num_frames=10, output_dir='calibration_frames'):
        """
        Extracts frames from the input video to use for camera calibration.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for i, frame_id in enumerate(frame_ids):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(output_dir, f'frame_{i}.jpg')
                cv2.imwrite(frame_path, frame)
                frames.append(frame)

        cap.release()
        return frames

    def calibrate(self, frames):
        """
        Calibrates the camera using the extracted frames.
        """
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

        if self.objpoints and self.imgpoints:
            ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            return ret

        return False

    def undistort(self, img):
        if self.mtx is not None and self.dist is not None:
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return img
