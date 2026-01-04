import cv2
import numpy as np
from datetime import datetime

from dataclasses import dataclass
from typing import List, Tuple

from .reid_chromadb import ChromaDBVehicleReID
from .feature_extraction import VehicleDescriptor
from .vehicle_detection import VehicleDetector
from . import color

class Pipeline:
    def __init__(self,
                detector: VehicleDetector = None,
                descriptor: VehicleDescriptor = None,
                classifier: ChromaDBVehicleReID = None,
                db_path: str = "./chroma_vehicle_reid") -> None:

        self.detector = detector or VehicleDetector()
        self.descriptor = descriptor or VehicleDescriptor(model_type='osnet')
        self.classifier = classifier or ChromaDBVehicleReID(db_path=db_path)
        
        self.processed_cross_camera = set()

    def process(self, frame, camera_id, save_cross_camera_images=True):
        try:
            VEHICLE_LABELS = {0: 'motorcycle', 1: 'car', 2: 'truck', 3: 'bus'}
            
            detected_data = self.detector.track(frame, camera_id)
            if len(detected_data) == 1:
                detected_data = detected_data[0]
            
            final_img = frame.copy()
            current_timestamp = datetime.now().isoformat()
            
            for detected_box in detected_data.boxes:
                xyxy = detected_box.xyxy.squeeze().tolist()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(frame.shape[1], xmax)
                ymax = min(frame.shape[0], ymax)
                
                cls = int(detected_box.cls)
                conf = float(detected_box.conf)
                track_id = int(detected_box.id) if detected_box.id is not None else -1

                vehicle_img = frame[ymin:ymax, xmin:xmax, :]
                
                if vehicle_img.size == 0:
                    continue

                vehicle_features = self.descriptor.extract_feature(vehicle_img)
                vehicle_id = self.classifier.identify(
                    target=vehicle_features,
                    vehicle_type=cls,
                    confidence=conf,
                    do_update=True,
                    image=vehicle_img,
                    camera_id=camera_id,
                    track_id=track_id,
                    timestamp=current_timestamp
                )
                
                if save_cross_camera_images:
                    camera_matches = self.classifier.get_cross_camera_matches(vehicle_id, cls)
                    if len(camera_matches) > 1:
                        vehicle_key = (camera_id, vehicle_id)
                        print(f"CROSS-CAMERA MATCH! Vehicle ID: {vehicle_id} ({VEHICLE_LABELS.get(cls, f'class_{cls}')}) found in cameras: {list(camera_matches.keys())}")
                        
                        if vehicle_key not in self.processed_cross_camera:
                            self.classifier.save_cross_camera_images(vehicle_id, cls, camera_matches)
                            self.processed_cross_camera.add(vehicle_key)

                vehicle_type = VEHICLE_LABELS.get(cls, f'class_{cls}')
                label = f"ID:{vehicle_id} ({vehicle_type})"

                unique_color = color.create_unique_color(vehicle_id)
                cv2.rectangle(
                    img=final_img,
                    pt1=(xmin, ymin),
                    pt2=(xmax, ymax),
                    color=unique_color,
                    thickness=2,
                )
                cv2.putText(
                    img=final_img,
                    text=label,
                    org=(xmin, ymin - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=unique_color,
                    thickness=2,
                )
            
            return final_img
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame 
    
    def get_pipeline_statistics(self):
        return self.classifier.get_statistics()
    
    def save_vehicle_database(self):
        stats = self.get_pipeline_statistics()
        print("Current database statistics:")
        for vehicle_type, count in stats.items():
            if vehicle_type != 'total' and vehicle_type != 'max_ids':
                print(f"  {vehicle_type}: {count} embeddings")
        print(f"Total: {stats['total']} embeddings")
        
    def close(self):
        self.classifier.close()
        print("Pipeline closed successfully")

@dataclass
class DetectionPayload:
    camera_id: str
    track_id: int
    vehicle_type: int
    confidence: float
    timestamp: str
    bbox: Tuple[int, int, int, int]
    image: np.ndarray
    feature: np.ndarray         
    thumbnail: bytes 

class Pipeline_spark:
    def __init__(
        self,
        detector: VehicleDetector = None,
        descriptor: VehicleDescriptor = None,
    ) -> None:
        self.detector = detector or VehicleDetector()
        self.descriptor = descriptor or VehicleDescriptor(model_type="osnet")

    def process(
        self,
        frame: np.ndarray,
        camera_id: str,
    ) -> Tuple[np.ndarray, List[DetectionPayload]]:
            
        detected = self.detector.track(frame, camera_id)
        if len(detected) == 1:
            detected = detected[0]

        timestamp = datetime.now().isoformat()
        payloads: List[DetectionPayload] = []

        for box in detected.boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy.squeeze().tolist())
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(frame.shape[1], xmax)
            ymax = min(frame.shape[0], ymax)

            patch = frame[ymin:ymax, xmin:xmax]
            if patch.size == 0:
                continue

            embedding = self.descriptor.extract_feature(patch)
            success, encoded = cv2.imencode(".jpg", patch)
            if not success:
                continue

            payload = DetectionPayload(
                camera_id=camera_id,
                track_id=int(box.id) if box.id is not None else -1,
                vehicle_type=int(box.cls),
                confidence=float(box.conf),
                timestamp=timestamp,
                bbox=(xmin, ymin, xmax, ymax),
                image=patch,
                feature=embedding,
                thumbnail=encoded.tobytes(),
            )
            payloads.append(payload)

        return payloads
    def close(self):
        print("Pipeline closed successfully")
