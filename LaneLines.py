import cv2
import numpy as np
import matplotlib.image as mpimg

def hist(img):
    bottom_half = img[img.shape[0] // 2:, :]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        self.left_curve_img = mpimg.imread('turn-left.png')
        self.right_curve_img = mpimg.imread('turn-right.png')
        self.keep_straight_img = mpimg.imread('decision.png')
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50
        self.speed_limit = "60 km/h"

    def forward(self, img):
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        topleft = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)
        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def extract_features(self, img):
        self.img = img
        self.window_height = int(img.shape[0] // self.nwindows)
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        assert(len(img.shape) == 2)
        out_img = np.dstack((img, img, img))
        histogram = hist(img)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height // 2
        leftx, lefty, rightx, righty = [], [], [], []
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)
            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)
            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))
        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)
        print(f"Left points: {len(lefty)}, Right points: {len(righty)}")
        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))
        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))
        ploty = np.linspace(miny, maxy, img.shape[0])
        left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))
        lR, rR, pos = self.measure_curvature()
        return out_img

    def plot(self, out_img):
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature()
        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]
        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')
        if len(self.dir) > 10:
            self.dir.pop(0)
        direction = max(set(self.dir), key=self.dir.count)
        msg = "Keep Straight"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
        icon_img = self.keep_straight_img
        if direction == 'L':
            icon_img = self.left_curve_img
            msg = "Left turn"
        elif direction == 'R':
            icon_img = self.right_curve_img
            msg = "Right turn"
        lane_width = self.calculate_lane_width()
        lane_visibility = "Clear" if self.clear_visibility else "Poor"
        departure_warning = self.calculate_departure_warning(pos)
        
        # Define box properties
        box_width = 400
        box_height = 500
        black = (0, 25, 51)
        violet = (255, 0, 127)
        
        # Top left box
        top_left_box = out_img[:box_height, :box_width]
        top_left_box[:, :, :] = black
        
        # Top right box
        top_right_box = out_img[:box_height, -box_width:]
        top_right_box[:, :, :] = black
        
        # Draw the icon in the center of the top right box
        icon_y, icon_x = icon_img.shape[:2]
        icon_center_x = box_width // 2
        icon_center_y = box_height 
        icon_start_x = icon_center_x - icon_x // 2
        icon_start_y = icon_center_y - icon_y - 300
        if icon_img.shape[2] == 4:  # RGBA image
            alpha_icon = icon_img[:, :, 3] / 255.0
            for c in range(3):
                top_right_box[icon_start_y:icon_start_y + icon_y, icon_start_x:icon_start_x + icon_x, c] = \
                    alpha_icon * icon_img[:, :, c] + (1 - alpha_icon) * top_right_box[icon_start_y:icon_start_y + icon_y, icon_start_x:icon_start_x + icon_x, c]
        else:
            top_right_box[icon_start_y:icon_start_y + icon_y, icon_start_x:icon_start_x + icon_x] = icon_img
        
        # Prepare text and boxes for the top right corner
        top_right_texts = [
            {"text": msg, "pos": (box_width // 2, 240)},
            {"text": curvature_msg, "pos": (box_width // 2, 320)},
            {"text": f"Lane Width = {lane_width:.2f} m", "pos": (box_width // 2, 400)},
        ]

        for item in top_right_texts:
            text_size = cv2.getTextSize(item["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = item["pos"][0] - text_size[0] // 2
            text_y = item["pos"][1] + text_size[1] // 2
            cv2.putText(top_right_box, item["text"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, violet, 2, cv2.LINE_AA)

        # Prepare text and boxes for the top left corner
        top_left_texts = [
            {"text": "On the right track", "pos": (box_width // 2, 80)},
            {"text": f"{abs(pos):.2f}m away from center", "pos": (box_width // 2, 160)},
            {"text": f"Visibility: {lane_visibility}", "pos": (box_width // 2, 320)},
            {"text": f"Speed Limit: {self.speed_limit}", "pos": (box_width // 2, 400)},
            {"text": departure_warning, "pos": (box_width // 2, 480)},
        ]

        for item in top_left_texts:
            text_size = cv2.getTextSize(item["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = item["pos"][0] - text_size[0] // 2
            text_y = item["pos"][1] + text_size[1] // 2
            cv2.putText(top_left_box, item["text"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, violet, 2, cv2.LINE_AA)

        return out_img

    

    def measure_curvature(self):
        ym = 30 / 720
        xm = 3.7 / 700
        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym
        left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        xl = np.dot(self.left_fit, [700 ** 2, 700, 1])
        xr = np.dot(self.right_fit, [700 ** 2, 700, 1])
        pos = (1280 // 2 - (xl + xr) // 2) * xm
        return left_curveR, right_curveR, pos

    def calculate_lane_width(self):
        """ Calculate the width of the lane """
        if self.left_fit is not None and self.right_fit is not None:
            ploty = np.linspace(0, self.img.shape[0] - 1, self.img.shape[0])
            left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
            right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]
            lane_width = np.mean(right_fitx - left_fitx)
            return lane_width * 3.7 / 700  # Convert from pixels to meters
        else:
            return 0.0

    def calculate_departure_warning(self, pos):
        """ Generate a departure warning based on the vehicle's position """
        if pos > 0.5:
            return "Warning: Departing Right"
        elif pos < -0.5:
            return "Warning: Departing Left"
        else:
            return "Lane Centered"
