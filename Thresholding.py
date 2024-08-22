import cv2
import numpy as np

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

class Thresholding:
    """ This class is for extracting relevant pixels in an image.
    """
    def __init__(self):
        """ Init Thresholding."""
        pass

    def forward(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # S channel from HLS space
        s_channel = hls[:,:,2]
        s_thresh = threshold_rel(s_channel, 0.7, 1.0)
        
        # B channel from LAB space
        b_channel = lab[:,:,2]
        b_thresh = threshold_rel(b_channel, 0.8, 1.0)
        
        # Combination of S and B channel thresholds
        combined = np.zeros_like(s_channel)
        combined[(s_thresh == 255) | (b_thresh == 255)] = 255
        
        return combined
