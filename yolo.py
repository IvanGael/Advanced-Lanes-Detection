from ultralytics import YOLO

class CarDetection:
    def __init__(self, model_path='yolo11n'):
        self.model = YOLO(model_path)

    def detect_cars(self, img):
        results = self.model(img, classes=[2, 3, 5, 7])  # Cars, motorcycles, buses, trucks
        return results[0]

    def draw_boxes(self, results):
        annotated_img = results.plot()
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
