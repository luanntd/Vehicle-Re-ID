import cv2
import numpy as np
import torch
from ultralytics import YOLO

device = 0 if torch.cuda.is_available() else "cpu"

class VehicleDetector:
    def __init__(self, model_path: str = 'yolo11n.pt'):
        # Load YOLOv8 Model
        self.yolo = YOLO(model_path)
        # Vehicle classes in COCO dataset: car(2), motorcycle(3), bus(5), truck(7)
        self.vehicle_classes = [2, 3, 5, 7]

    def detect(self, input_bytes: bytes):
        """
        Detect vehicles in an image.

        Parameters
        ----------
        input_bytes: bytes
            Image stored in bytes format that needs vehicle detection.

        Returns
        -------
        Detection results containing bounding boxes and class information.
        """
        # Convert bytes to image
        image = cv2.imdecode(
            np.frombuffer(input_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        # Perform vehicle detection
        results = self.yolo(image, classes=self.vehicle_classes, device=device, conf=0.5)
        return results
