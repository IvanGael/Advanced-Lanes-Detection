import cvzone
from ultralytics import YOLO
import numpy as np
import cv2

class CarDetection:
    def __init__(self, model_path='yolo11n'):
        self.model = YOLO(model_path)

    def detect_cars(self, img):
        results = self.model(img, classes=[2, 3, 5, 7])  # Cars, motorcycles, buses, trucks
        return results[0]

    def draw_boxes(self, results):
        annotated_img = results.plot()
        distances = self.get_vehicle_distances(results)
        centroids = self.get_vehicle_centroids(results)
        
        for i, (dist, centroid) in enumerate(zip(distances, centroids)):
            bbox = results.boxes[i].xyxy[0]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cvzone.putTextRect(annotated_img, f"{dist:.2f}m", (x1, y1 - 40), scale=1, thickness=2, colorR=(159, 7, 247), offset=10)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (159, 7, 247), 2)
        
        return annotated_img

    def get_vehicle_distances(self, results, camera_height=1.5):
        distances = []
        for r in results.boxes:
            bbox = r.xyxy[0]
            # Distance estimation using bounding box height
            bbox_height = bbox[3] - bbox[1]
            distance = (camera_height * 720) / bbox_height  
            distances.append(distance)
        return distances

    def get_vehicle_centroids(self, results):
        centroids = []
        for r in results.boxes:
            bbox = r.xyxy[0]
            centroid_x = (bbox[0] + bbox[2]) / 2
            centroid_y = (bbox[1] + bbox[3]) / 2
            centroids.append((centroid_x, centroid_y))
        return centroids

    def calculate_pairwise_distances(self, centroids):
        pairwise_distances = []
        for i, c1 in enumerate(centroids):
            for j, c2 in enumerate(centroids):
                if i != j:
                    distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    pairwise_distances.append((i, j, distance))
        return pairwise_distances
