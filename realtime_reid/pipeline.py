import os
import cv2
import numpy as np

from .classifier import VehicleReID
from .feature_extraction import VehicleDescriptor
from .vehicle_detector import VehicleDetector
from .visualization_utils import color

class Pipeline:
    def __init__(self,
                detector: VehicleDetector = None,
                descriptor: VehicleDescriptor = None,
                classifier: VehicleReID = None) -> None:
        """Initialize the pipeline with vehicle detection and re-identification components."""
        self.detector = detector or VehicleDetector()
        self.descriptor = descriptor or VehicleDescriptor(model_type='osnet')
        self.classifier = classifier or VehicleReID()
        # Persistent dictionary for all vehicle types
        self.vehicle_by_type = {
            0: [],  # motorcycle
            1: [],  # car
            2: [],  # truck
            3: []   # bus
        }

    def process(
        self,
        msg,
        tag,
        save_dir: str = None,
        return_bytes: bool = False
    ):
        """
        Process the input message by detecting and identifying vehicles in the image.

        Parameters
        ----------
        msg: bytes | np.ndarray
            The input message containing the image data.
        save_dir: str, optional
            Directory to save detected vehicle images.
        return_bytes: bool
            Whether to return the processed image as bytes.

        Returns
        -------
        Processed image with bounding boxes and vehicle IDs.
        """
        if not isinstance(msg, bytes) and not isinstance(msg, np.ndarray):
            raise TypeError(f"msg must be bytes or numpy array. Got {type(msg)}.")

        VEHICLE_LABELS = {0: 'motorcycle', 1: 'car', 2: 'truck', 3: 'bus'}

        detected_data = self.detector.track(msg, camera_id=tag)
        if len(detected_data) == 1:
            detected_data = detected_data[0]

        # Convert image data to array
        image_data = np.frombuffer(msg, dtype=np.uint8)
        final_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        for detected_box in detected_data.boxes:
            # Get bounding box coordinates
            xyxy = detected_box.xyxy.squeeze().tolist()
            xmin, ymin, xmax, ymax = map(int, xyxy)
            
            # Get detected class (vehicle type)
            cls = int(detected_box.cls)
            conf = float(detected_box.conf)
            track_id = int(detected_box.id) if detected_box.id is not None else -1

            # Crop vehicle image
            vehicle_img = final_img[ymin:ymax, xmin:xmax, :]

            # Extract features and identify vehicle
            vehicle_features = self.descriptor.extract_feature(vehicle_img)
            vehicle_id = self.classifier.identify(
                target=vehicle_features,
                vehicle_type=cls,
                confidence=conf,
                do_update=True
            )
            
            # Store vehicle info by type, including tag
            self.vehicle_by_type[cls].append({'id': vehicle_id, 'image': vehicle_img, 'tag': tag, 'track_id': track_id})

            # Save cropped vehicle image if directory provided
            # if save_dir:
            #     save_filename = f"{len(os.listdir(save_dir))}_{vehicle_id}_{cls}"
            #     cv2.imwrite(
            #         f"{save_dir}/{save_filename}.jpg",
            #         vehicle_img
            #     )
            
            # Save matching images if duplicate vehicle_id found (only for same vehicle type and different tag)
            vehicle_type = VEHICLE_LABELS.get(cls, f'class_{cls}')
            matches = [entry for entry in self.vehicle_by_type[cls] if entry['id'] == vehicle_id]
            tags = set(entry['tag'] for entry in matches)
            if len(tags) > 1:
                print(f"Found cross-camera matching vehicle ID: {vehicle_id} ({vehicle_type})")
                vehicle_dir = f'{vehicle_type}_{vehicle_id}'
                match_dir = os.path.join('matching', vehicle_dir)
                if not os.path.exists(match_dir):
                    os.makedirs(match_dir)
                # Only save one image per unique track_id per camera
                saved = set()
                for entry in matches:
                    key = (entry['tag'], entry['track_id'])
                    if key not in saved:
                        cv2.imwrite(os.path.join(match_dir, f"{entry['tag']}_track{entry['track_id']}.jpg"), entry['image'])
                        saved.add(key)


            # Create label with vehicle ID and type
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

        if return_bytes:
            return cv2.imencode(
                '.jpg',
                final_img,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )[1].tobytes()

        return final_img
