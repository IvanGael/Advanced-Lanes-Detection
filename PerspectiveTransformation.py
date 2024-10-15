import cv2
import numpy as np

class PerspectiveTransformation:
    """ This a class for transforming image between front view and top view

    Attributes:
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        M (np.array): Matrix to transform image from front view to top view
        M_inv (np.array): Matrix to transform image from top view to front view
    """
    def __init__(self):
        self.src = np.float32([
            [0.43, 0.65],
            [0.58, 0.65],
            [0.1, 1],
            [0.9, 1]
        ])
        self.dst = np.float32([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1]
        ])
        self.img_size = None

    def update_matrices(self, img_size):
        self.img_size = img_size
        src_scaled = self.src * np.float32([img_size[0], img_size[1]])
        dst_scaled = self.dst * np.float32([img_size[0], img_size[1]])
        self.M = cv2.getPerspectiveTransform(src_scaled, dst_scaled)
        self.M_inv = cv2.getPerspectiveTransform(dst_scaled, src_scaled)

    def forward(self, img, flags=cv2.INTER_LINEAR):
        if self.img_size != img.shape[:2][::-1]:
            self.update_matrices(img.shape[:2][::-1])
        return cv2.warpPerspective(img, self.M, self.img_size, flags=flags)

    def backward(self, img, flags=cv2.INTER_LINEAR):
        if self.img_size != img.shape[:2][::-1]:
            self.update_matrices(img.shape[:2][::-1])
        return cv2.warpPerspective(img, self.M_inv, self.img_size, flags=flags)
    
    def get_aerial_view(self, img):
        """Generate a top-down (bird's-eye) view of the lane."""
        h, w = img.shape[:2]
        src_points = np.float32([[w*0.45, h*0.65], [w*0.55, h*0.65], [w*0.9, h], [w*0.1, h]])
        dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        top_view = cv2.warpPerspective(img, M, (w, h))
        return top_view

    def overlay_aerial_view(self, img, aerial_img, scale=0.3):
        """Overlay aerial view at top center."""
        h, w = img.shape[:2]
        aerial_img = cv2.resize(aerial_img, (int(w * scale), int(h * scale)))

        # Define where to overlay (top center)
        x_offset = (w - aerial_img.shape[1]) // 2
        y_offset = 0  # Top center

        img[y_offset:y_offset+aerial_img.shape[0], x_offset:x_offset+aerial_img.shape[1]] = aerial_img
        return img
