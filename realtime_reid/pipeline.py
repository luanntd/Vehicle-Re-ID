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
        self.descriptor = descriptor or VehicleDescriptor()
        self.classifier = classifier or VehicleReID()

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

        detected_data = self.detector.detect(msg)
        if len(detected_data) == 1:
            detected_data = detected_data[0]

        # Convert image data to array
        image_data = np.frombuffer(msg, dtype=np.uint8)
        final_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        vehicle_ids = []
        vehicle_images = []
        for detected_box in detected_data.boxes:
            # Get bounding box coordinates
            xyxy = detected_box.xyxy.squeeze().tolist()
            xmin, ymin, xmax, ymax = map(int, xyxy)
            
            # Get detected class (vehicle type)
            cls = int(detected_box.cls)
            conf = float(detected_box.conf)
            
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
            vehicle_ids.append(vehicle_id)
            vehicle_images.append(vehicle_img)

            # Save cropped vehicle image if directory provided
            if save_dir:
                save_filename = f"{len(os.listdir(save_dir))}_{vehicle_id}_{cls}"
                cv2.imwrite(
                    f"{save_dir}/{save_filename}.jpg",
                    vehicle_img
                )
            
            # Save matching images if duplicate vehicle_id found
            if tag == 'cam2':
                if vehicle_ids.count(vehicle_id) > 1:
                    print(f"Found matching vehicle ID: {vehicle_id}")
                    vehicle_dir = str(f'{cls}_' + str(vehicle_id))
                    print(vehicle_dir)
                    print(type(vehicle_dir))
                    match_dir = os.path.join('matching', vehicle_dir)
                    if not os.path.exists(match_dir):
                        os.makedirs(match_dir)
                    # Save all images with this id
                    for jdx, vimg in enumerate(vehicle_images):
                        if vehicle_ids[jdx] == vehicle_id:
                            cv2.imwrite(os.path.join(match_dir, f'{vehicle_id}_{jdx}.jpg'), vimg)

        # Draw bounding boxes and labels
        COCO_VEHICLE_LABELS = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        for detected_box, vehicle_id in zip(detected_data.boxes, vehicle_ids):
            xyxy = detected_box.xyxy.squeeze().tolist()
            xmin, ymin, xmax, ymax = map(int, xyxy)
            cls = int(detected_box.cls)

            # Create label with vehicle ID and type
            vehicle_type = COCO_VEHICLE_LABELS.get(cls, f'class_{cls}')
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
