# import os
# import cv2
# import numpy as np

# from .classifier import VehicleReID
# from .feature_extraction import VehicleDescriptor
# from .vehicle_detector import VehicleDetector
# from .visualization_utils import color

# class Pipeline:
#     def __init__(self,
#                 detector: VehicleDetector = None,
#                 descriptor: VehicleDescriptor = None,
#                 classifier: VehicleReID = None) -> None:
#         """Initialize the pipeline with vehicle detection and re-identification components."""
#         self.detector = detector or VehicleDetector()
#         self.descriptor = descriptor or VehicleDescriptor(model_type='osnet')
#         self.classifier = classifier or VehicleReID()
#         # Persistent dictionary for all vehicle types
#         self.vehicle_by_type = {
#             0: [],  # motorcycle
#             1: [],  # car
#             2: [],  # truck
#             3: []   # bus
#         }

#     # def process(
#     #     self,
#     #     msg,
#     #     tag,
#     #     save_dir: str = None,
#     #     return_bytes: bool = False
#     # ):
#     #     """
#     #     Process the input message by detecting and identifying vehicles in the image.

#     #     Parameters
#     #     ----------
#     #     msg: bytes | np.ndarray
#     #         The input message containing the image data.
#     #     save_dir: str, optional
#     #         Directory to save detected vehicle images.
#     #     return_bytes: bool
#     #         Whether to return the processed image as bytes.

#     #     Returns
#     #     -------
#     #     Processed image with bounding boxes and vehicle IDs.
#     #     """
#     #     if not isinstance(msg, bytes) and not isinstance(msg, np.ndarray):
#     #         raise TypeError(f"msg must be bytes or numpy array. Got {type(msg)}.")

#     #     VEHICLE_LABELS = {0: 'motorcycle', 1: 'car', 2: 'truck', 3: 'bus'}

#     #     detected_data = self.detector.track(msg, camera_id=tag)
#     #     if len(detected_data) == 1:
#     #         detected_data = detected_data[0]

#     #     # Convert image data to array
#     #     image_data = np.frombuffer(msg, dtype=np.uint8)
#     #     final_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
#     #     for detected_box in detected_data.boxes:
#     #         # Get bounding box coordinates
#     #         xyxy = detected_box.xyxy.squeeze().tolist()
#     #         xmin, ymin, xmax, ymax = map(int, xyxy)
            
#     #         # Get detected class (vehicle type)
#     #         cls = int(detected_box.cls)
#     #         conf = float(detected_box.conf)
#     #         track_id = int(detected_box.id) if detected_box.id is not None else -1

#     #         # Crop vehicle image
#     #         vehicle_img = final_img[ymin:ymax, xmin:xmax, :]

#     #         # Extract features and identify vehicle
#     #         vehicle_features = self.descriptor.extract_feature(vehicle_img)
#     #         vehicle_id = self.classifier.identify(
#     #             target=vehicle_features,
#     #             vehicle_type=cls,
#     #             confidence=conf,
#     #             do_update=True
#     #         )
            
#     #         # Store vehicle info by type, including tag
#     #         self.vehicle_by_type[cls].append({'id': vehicle_id, 'image': vehicle_img, 'tag': tag, 'track_id': track_id})

#     #         # Save cropped vehicle image if directory provided
#     #         # if save_dir:
#     #         #     save_filename = f"{len(os.listdir(save_dir))}_{vehicle_id}_{cls}"
#     #         #     cv2.imwrite(
#     #         #         f"{save_dir}/{save_filename}.jpg",
#     #         #         vehicle_img
#     #         #     )
            
#     #         # Save matching images if duplicate vehicle_id found (only for same vehicle type and different tag)
#     #         vehicle_type = VEHICLE_LABELS.get(cls, f'class_{cls}')
#     #         matches = [entry for entry in self.vehicle_by_type[cls] if entry['id'] == vehicle_id]
#     #         tags = set(entry['tag'] for entry in matches)
#     #         if len(tags) > 1:
#     #             print(f"Found cross-camera matching vehicle ID: {vehicle_id} ({vehicle_type})")
#     #             vehicle_dir = f'{vehicle_type}_{vehicle_id}'
#     #             match_dir = os.path.join('matching', vehicle_dir)
#     #             if not os.path.exists(match_dir):
#     #                 os.makedirs(match_dir)
#     #             # Only save one image per unique track_id per camera
#     #             saved = set()
#     #             for entry in matches:
#     #                 key = (entry['tag'], entry['track_id'])
#     #                 if key not in saved:
#     #                     cv2.imwrite(os.path.join(match_dir, f"{entry['tag']}_track{entry['track_id']}.jpg"), entry['image'])
#     #                     saved.add(key)


#     #         # Create label with vehicle ID and type
#     #         label = f"ID:{vehicle_id} ({vehicle_type})"

#     #         # Draw bounding box and label
#     #         unique_color = color.create_unique_color(vehicle_id)
#     #         cv2.rectangle(
#     #             img=final_img,
#     #             pt1=(xmin, ymin),
#     #             pt2=(xmax, ymax),
#     #             color=unique_color,
#     #             thickness=2,
#     #         )
#     #         cv2.putText(
#     #             img=final_img,
#     #             text=label,
#     #             org=(xmin, ymin - 5),
#     #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#     #             fontScale=0.6,
#     #             color=unique_color,
#     #             thickness=2,
#     #         )

#     #     if return_bytes:
#     #         return cv2.imencode(
#     #             '.jpg',
#     #             final_img,
#     #             [cv2.IMWRITE_JPEG_QUALITY, 100]
#     #         )[1].tobytes()

#     #     return final_img

#     def process(self, frame, camera_id):
#         """
#         Process a frame directly without encoding/decoding to save memory.
        
#         Parameters
#         ----------
#         frame: np.ndarray
#             OpenCV frame (BGR format)
#         camera_id: str
#             Camera identifier
            
#         Returns
#         -------
#         np.ndarray: Processed frame with bounding boxes and vehicle IDs
#         """
#         try:
#             VEHICLE_LABELS = {0: 'motorcycle', 1: 'car', 2: 'truck', 3: 'bus'}
            
#             # Use direct frame tracking to avoid encoding/decoding
#             detected_data = self.detector.track(frame, camera_id)
#             if len(detected_data) == 1:
#                 detected_data = detected_data[0]
            
#             # Work directly with the input frame
#             final_img = frame.copy()
            
#             for detected_box in detected_data.boxes:
#                 # Get bounding box coordinates
#                 xyxy = detected_box.xyxy.squeeze().tolist()
#                 xmin, ymin, xmax, ymax = map(int, xyxy)
                
#                 # Ensure coordinates are within frame bounds
#                 xmin = max(0, xmin)
#                 ymin = max(0, ymin)
#                 xmax = min(frame.shape[1], xmax)
#                 ymax = min(frame.shape[0], ymax)
                
#                 # # Skip if bounding box is too small
#                 # if xmax - xmin < 20 or ymax - ymin < 20:
#                 #     continue
                
#                 # Get detected class (vehicle type)
#                 cls = int(detected_box.cls)
#                 conf = float(detected_box.conf)
#                 track_id = int(detected_box.id) if detected_box.id is not None else -1

#                 # Crop vehicle image
#                 vehicle_img = final_img[ymin:ymax, xmin:xmax, :]
                
#                 if vehicle_img.size == 0:
#                     continue

#                 # Extract features and identify vehicle
#                 vehicle_features = self.descriptor.extract_feature(vehicle_img)
#                 vehicle_id = self.classifier.identify(
#                     target=vehicle_features,
#                     vehicle_type=cls,
#                     confidence=conf,
#                     do_update=True
#                 )
                
#                 # Store vehicle info by type, including tag
#                 self.vehicle_by_type[cls].append({
#                     'id': vehicle_id, 
#                     'image': vehicle_img, 
#                     'tag': camera_id, 
#                     'track_id': track_id
#                 })

#                 # Check for cross-camera matches
#                 vehicle_type = VEHICLE_LABELS.get(cls, f'class_{cls}')
#                 matches = [entry for entry in self.vehicle_by_type[cls] if entry['id'] == vehicle_id]
#                 tags = set(entry['tag'] for entry in matches)
#                 if len(tags) > 1:
#                     print(f"Found cross-camera matching vehicle ID: {vehicle_id} ({vehicle_type})")
#                     vehicle_dir = f'{vehicle_type}_{vehicle_id}'
#                     match_dir = os.path.join('matching', vehicle_dir)
#                     if not os.path.exists(match_dir):
#                         os.makedirs(match_dir)
#                     # Only save one image per unique track_id per camera
#                     saved = set()
#                     for entry in matches:
#                         key = (entry['tag'], entry['track_id'])
#                         if key not in saved:
#                             cv2.imwrite(os.path.join(match_dir, f"{entry['tag']}_track{entry['track_id']}.jpg"), entry['image'])
#                             saved.add(key)

#                 # Create label with vehicle ID and type
#                 label = f"ID:{vehicle_id} ({vehicle_type})"

#                 # Draw bounding box and label
#                 unique_color = color.create_unique_color(vehicle_id)
#                 cv2.rectangle(
#                     img=final_img,
#                     pt1=(xmin, ymin),
#                     pt2=(xmax, ymax),
#                     color=unique_color,
#                     thickness=2,
#                 )
#                 cv2.putText(
#                     img=final_img,
#                     text=label,
#                     org=(xmin, ymin - 5),
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.6,
#                     color=unique_color,
#                     thickness=2,
#                 )
            
#             return final_img
            
#         except Exception as e:
#             print(f"Error processing frame: {e}")
#             return frame  # Return original frame if processing fails





import os
import cv2
import numpy as np
from datetime import datetime

from .classifier_chromadb import ChromaDBVehicleReID
from .feature_extraction import VehicleDescriptor
from .vehicle_detector import VehicleDetector
from .visualization_utils import color

class Pipeline:
    def __init__(self,
                detector: VehicleDetector = None,
                descriptor: VehicleDescriptor = None,
                classifier: ChromaDBVehicleReID = None,
                db_path: str = "./chroma_vehicle_reid") -> None:
        """Initialize the pipeline with vehicle detection and re-identification components."""
        self.detector = detector or VehicleDetector()
        self.descriptor = descriptor or VehicleDescriptor(model_type='osnet')
        self.classifier = classifier or ChromaDBVehicleReID(db_path=db_path)
        
        # Track processed vehicles to avoid duplicate cross-camera saves
        self.processed_cross_camera = set()

    def process(self, frame, camera_id, save_cross_camera_images=True):
        """
        Process a frame directly without encoding/decoding to save memory.
        
        Parameters
        ----------
        frame: np.ndarray
            OpenCV frame (BGR format)
        camera_id: str
            Camera identifier
        save_cross_camera_images: bool
            Whether to save cross-camera matching images
            
        Returns
        -------
        np.ndarray: Processed frame with bounding boxes and vehicle IDs
        """
        try:
            VEHICLE_LABELS = {0: 'motorcycle', 1: 'car', 2: 'truck', 3: 'bus'}
            
            # Use direct frame tracking to avoid encoding/decoding
            detected_data = self.detector.track(frame, camera_id)
            if len(detected_data) == 1:
                detected_data = detected_data[0]
            
            # Work directly with the input frame
            final_img = frame.copy()
            current_timestamp = datetime.now().isoformat()
            
            for detected_box in detected_data.boxes:
                # Get bounding box coordinates
                xyxy = detected_box.xyxy.squeeze().tolist()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                
                # Ensure coordinates are within frame bounds
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(frame.shape[1], xmax)
                ymax = min(frame.shape[0], ymax)
                
                # Get detected class (vehicle type)
                cls = int(detected_box.cls)
                conf = float(detected_box.conf)
                track_id = int(detected_box.id) if detected_box.id is not None else -1

                # Crop vehicle image
                vehicle_img = frame[ymin:ymax, xmin:xmax, :]
                
                if vehicle_img.size == 0:
                    continue

                # Extract features and identify vehicle using ChromaDB
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
                
                # Check for cross-camera matches and save images
                if save_cross_camera_images:
                    # Always check for cross-camera matches (not just for new vehicles)
                    camera_matches = self.classifier.get_cross_camera_matches(vehicle_id, cls)
                    if len(camera_matches) > 1:
                        # Use (camera_id, track_id) as key to ensure uniqueness per camera/track combination
                        # This prevents duplicate image saving for the same track in a camera
                        vehicle_key = (camera_id, track_id)
                        print(f"CROSS-CAMERA MATCH! Vehicle ID: {vehicle_id} ({VEHICLE_LABELS.get(cls, f'class_{cls}')}) found in cameras: {list(camera_matches.keys())}")
                        
                        # Only save images once per camera/track combination to avoid duplicates
                        if vehicle_key not in self.processed_cross_camera:
                            self.classifier.save_cross_camera_images(vehicle_id, cls)
                            self.processed_cross_camera.add(vehicle_key)
                            # print(f"📸 Saved cross-camera images for vehicle {vehicle_id} (camera {camera_id}, track {track_id})")

                # Create label with vehicle ID and type
                vehicle_type = VEHICLE_LABELS.get(cls, f'class_{cls}')
                label = f"ID:{vehicle_id} ({vehicle_type})"

                # Draw bounding box and label
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
            return frame  # Return original frame if processing fails
    
    def get_pipeline_statistics(self):
        """Get statistics about the pipeline's performance."""
        return self.classifier.get_statistics()
    
    def save_vehicle_database(self):
        """Save the current state of the vehicle database."""
        # ChromaDB automatically persists data, but we can get stats
        stats = self.get_pipeline_statistics()
        print("Current database statistics:")
        for vehicle_type, count in stats.items():
            if vehicle_type != 'total' and vehicle_type != 'max_ids':
                print(f"  {vehicle_type}: {count} embeddings")
        print(f"  Total: {stats['total']} embeddings")
        
    def close(self):
        """Close pipeline and database connections."""
        self.classifier.close()
        print("Pipeline closed successfully")

# Utility function for Spark integration
def create_spark_compatible_pipeline(db_path: str = "./chroma_vehicle_reid"):
    """
    Create a pipeline instance that's compatible with Spark serialization.
    
    Parameters
    ----------
    db_path: str
        Path to ChromaDB database
        
    Returns
    -------
    Pipeline: Configured pipeline instance
    """
    detector = VehicleDetector(model_path='checkpoints/best_20.pt')
    descriptor = VehicleDescriptor(
        model_type='osnet', 
        model_path='checkpoints/best_osnet_model.pth'
    )
    classifier = ChromaDBVehicleReID(db_path=db_path)
    
    return Pipeline(
        detector=detector,
        descriptor=descriptor,
        classifier=classifier,
        db_path=db_path
    )

# Function for processing frames in Spark
def process_frame_spark(frame_data, camera_id, db_path="./chroma_vehicle_reid"):
    """
    Process a single frame in Spark context.
    This function creates its own pipeline instance to avoid serialization issues.
    
    Parameters
    ----------
    frame_data: bytes or np.ndarray
        Frame data
    camera_id: str
        Camera identifier
    db_path: str
        Path to ChromaDB database
        
    Returns
    -------
    np.ndarray: Processed frame
    """
    # Create pipeline instance for this Spark task
    pipeline = create_spark_compatible_pipeline(db_path)
    
    try:
        # Convert bytes to frame if needed
        if isinstance(frame_data, bytes):
            frame = cv2.imdecode(
                np.frombuffer(frame_data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
        else:
            frame = frame_data
        
        # Process frame
        result = pipeline.process(frame, camera_id)
        
        # Close pipeline
        pipeline.close()
        
        return result
        
    except Exception as e:
        print(f"Error in Spark frame processing: {e}")
        pipeline.close()
        return frame_data if isinstance(frame_data, np.ndarray) else None
