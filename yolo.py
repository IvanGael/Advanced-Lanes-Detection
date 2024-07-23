# yolo.py

from ultralytics import YOLO
import numpy as np

class CarDetection:
    def __init__(self, model_path='yolov8n'):
        self.model = YOLO(model_path)

    def detect_cars(self, img):
        results = self.model(img, classes=[2, 3, 5, 7])  # 2: car, 3: motorcycle, 5: bus, 7: truck
        return results[0]

    def draw_boxes(self, img, results):
        annotated_img = results.plot()
        return annotated_img

    def process_image(self, img):
        results = self.detect_cars(img)
        annotated_img = self.draw_boxes(img, results)
        return annotated_img