import os
import time
import cv2
from kafka import KafkaProducer
import tqdm
class VideoProducer:
    def __init__(
            self,
            topic: str,
            interval: float,
            bootstrap_servers: str = 'localhost:9092'):
        
        self.INTERVAL = interval
        # Convert FPS to interval if needed
        if self.INTERVAL > 1:
            self.INTERVAL = 1 / self.INTERVAL

        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        self.TOPIC = topic

    def encode_and_produce(self, frame, interval: float, frame_count: int = 0):
        """Process and send a single frame to Kafka."""
        # Only send every 5th frame
        if frame_count % 5 != 0:
            return
            
        frame = self.process_frame(frame)

        # Convert frame to jpg format
        _, buffer = cv2.imencode('.jpg', frame)

        # Send to Kafka
        self.producer.send(self.TOPIC, buffer.tobytes())

        time.sleep(interval)

    def publish_video(self, source: str):
        """
        Publish video frames to Kafka topic.
        
        Parameters
        ----------
        source: str
            Path to video file or directory of images
        """
        try:
            if os.path.isfile(source):
                print(f"Publishing video {source} to topic {self.TOPIC}")
                self._publish_from_video(source)
            else:
                print(f"Publishing from image folder {source} to topic {self.TOPIC}")
                self._publish_from_images(source)

            print('Publishing complete!')
        except KeyboardInterrupt:
            print("Publishing stopped.")

    def _publish_from_video(self, video_path: str):
        """Publish frames from a video file."""
        video = cv2.VideoCapture(video_path)
        progress_bar = tqdm.tqdm(
            total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            desc=f"Publishing {os.path.basename(video_path)}",
            unit="frames"
        )
        # Use video FPS if no interval specified
        if self.INTERVAL == -1:
            self.INTERVAL = 1 / video.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            self.encode_and_produce(frame, self.INTERVAL, frame_count)
            frame_count += 1
            progress_bar.update(1)

        video.release()

    def _publish_from_images(self, image_dir: str):
        """Publish frames from a directory of images."""
        image_files = [f for f in os.listdir(image_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Default to 12 FPS for image sequences
        if self.INTERVAL == -1:
            self.INTERVAL = 1 / 12

        for img_file in sorted(image_files):
            image_path = os.path.join(image_dir, img_file)
            frame = cv2.imread(image_path)
            self.encode_and_produce(frame, self.INTERVAL)

    @staticmethod
    def process_frame(frame):
        """Preprocess frame before sending."""
        # Resize while maintaining aspect ratio
        target_width = 1280  # Standard HD width
        aspect_ratio = frame.shape[1] / frame.shape[0]
        target_height = int(target_width / aspect_ratio)
        
        frame = cv2.resize(frame, (target_width, target_height))
        return frame
