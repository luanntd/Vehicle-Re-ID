import cv2
import numpy as np
import torch
from ultralytics import YOLO

device = 0 if torch.cuda.is_available() else "cpu"

class VehicleDetector:
    def __init__(self, model_path: str = 'best_20.pt'):
        # Load Model path
        self.model_path = model_path
        # Vehicle classes: motorcycle(0), car(1), truck(2), bus(3) 
        self.vehicle_classes = [0, 1, 2, 3]
        # Separate tracker instances for each camera
        self.camera_trackers = {}

    def detect(self, input_bytes: bytes):
        """
        Detect vehicles in an image (single frame detection).
        
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
        results = self.yolo(image, classes=self.vehicle_classes, device=device, conf=0.35)
        return results

    # def track(self, input_bytes: bytes, camera_id: str, tracker: str = "bytetrack.yaml"):
    #     """
    #     Track vehicles in a single camera feed.
        
    #     Parameters
    #     ----------
    #     frame_bytes: bytes
    #         Frame from specific camera
    #     camera_id: str
    #         Unique identifier for the camera (e.g., "cam_1", "cam_2")
    #     timestamp: float
    #         Timestamp of the frame (for temporal reasoning)
            
    #     Returns
    #     -------
    #     dict: Contains local tracking results and global vehicle IDs
    #     """
            
    #     # Convert bytes to image
    #     image = cv2.imdecode(
    #         np.frombuffer(input_bytes, np.uint8),
    #         cv2.IMREAD_COLOR
    #     )

    #     # Create separate tracker instance for each camera
    #     if camera_id not in self.camera_trackers:
    #         self.camera_trackers[camera_id] = YOLO(self.model_path)

    #     # Track vehicles in this specific camera
    #     results = self.camera_trackers[camera_id].track(
    #         image,
    #         classes=self.vehicle_classes,
    #         device=device,
    #         conf=0.35,
    #         tracker=tracker,
    #         persist=True,
    #         verbose=False
    #     )

    #     return results
    
    def track(self, image, camera_id, tracker: str = "bytetrack.yaml"):
        """
        Track vehicles in a frame directly without byte conversion.
        
        Parameters
        ----------
        image: np.ndarray
            OpenCV image array (BGR format)
        camera_id: str
            Camera identifier for tracking persistence
        timestamp: float
            Timestamp of the frame (for temporal reasoning)
            
        Returns
        -------
        dict: Contains local tracking results and global vehicle IDs
        """
        # Create separate tracker instance for each camera
        if camera_id not in self.camera_trackers:
            self.camera_trackers[camera_id] = YOLO(self.model_path)

        # Track vehicles in this specific camera
        results = self.camera_trackers[camera_id].track(
            image,
            classes=self.vehicle_classes,
            device=device,
            conf=0.35,
            tracker=tracker,
            persist=True,
            verbose=False
        )

        return results