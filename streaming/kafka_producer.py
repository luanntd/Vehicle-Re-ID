import os
import time
import cv2
import threading
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import tqdm
import numpy as np
from typing import Optional, Union
import gc


class VideoProducer:
    """
    Unified Video Producer for Kafka streaming.
    Supports both real-time streaming and batch processing modes.
    Compatible with both Producer.py and app.py usage patterns.
    """
    
    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = 'localhost:9092',
        fps: Optional[float] = None,
        interval: Optional[float] = None,
        mode: str = 'streaming'  # 'streaming' or 'batch'
    ):
        """
        Initialize VideoProducer.
        
        Args:
            topic (str): Kafka topic name
            bootstrap_servers (str): Kafka bootstrap servers
            fps (float, optional): Target FPS for streaming. If None, uses video's original FPS
            interval (float, optional): Frame interval in seconds. If > 1, treated as FPS
            mode (str): 'streaming' for real-time or 'batch' for batch processing
        """
        self.topic_name = topic
        self.bootstrap_servers = bootstrap_servers
        self.mode = mode
        self.producer = None
        self.is_streaming = False
        
        # Handle FPS/interval conversion for backward compatibility
        if interval is not None and interval != -1:
            if interval > 1:
                # Treat as FPS
                self.fps = interval
                self.interval = 1.0 / interval
            elif interval > 0:
                # Treat as interval
                self.interval = interval
                self.fps = 1.0 / interval
            else:
                # Invalid interval, use defaults
                self.fps = None
                self.interval = None
        elif fps is not None and fps > 0:
            self.fps = fps
            self.interval = 1.0 / fps
        else:
            # Use defaults - will be set later based on video properties
            self.fps = None
            self.interval = None
        
        # Create topic if it doesn't exist
        self._create_topic()
    
    def _create_topic(self):
        """Create Kafka topic if it doesn't exist."""
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                client_id='video_producer'
            )
            
            # Check if topic already exists
            existing_topics = admin_client.list_topics()
            if self.topic_name in existing_topics:
                print(f"[Producer] Topic {self.topic_name} already exists")
                admin_client.close()
                return
            
            # Create new topic
            topic_list = [NewTopic(
                name=self.topic_name,
                num_partitions=1,
                replication_factor=1
            )]
            
            admin_client.create_topics(new_topics=topic_list, validate_only=False)
            print(f"[Producer] Created topic: {self.topic_name}")
            admin_client.close()
            
        except Exception as e:
            print(f"[Producer] Topic creation info: {e}")
    
    def _get_producer_config(self):
        """Get Kafka producer configuration based on mode."""
        if self.mode == 'streaming':
            # Optimized for real-time streaming
            return {
                'bootstrap_servers': self.bootstrap_servers,
                'max_request_size': 10485760,  # 10MB
                'buffer_memory': 67108864,     # 64MB
                'max_in_flight_requests_per_connection': 5,
                'batch_size': 16384,           # 16KB
                'linger_ms': 10,               # Small batching delay
                'acks': 1,                     # Wait for one broker
                'compression_type': 'gzip',
                'value_serializer': None,
                'key_serializer': None,
            }
        else:
            # Optimized for batch processing
            return {
                'bootstrap_servers': self.bootstrap_servers,
                'max_request_size': 10485760,  # 10MB
                'buffer_memory': 33554432,     # 32MB
                'batch_size': 32768,           # 32KB
                'linger_ms': 5,
                'acks': 1,
                'compression_type': 'gzip',
                'value_serializer': None,
                'key_serializer': None,
            }
    
    def start_streaming(self, video_path: str):
        """
        Start streaming video frames to Kafka (streaming mode).
        Compatible with app.py usage.
        
        Args:
            video_path (str): Path to video file
        """
        if self.is_streaming:
            print(f"[Producer] Already streaming to {self.topic_name}")
            return
        
        self.is_streaming = True
        
        try:
            # Initialize Kafka producer
            producer_config = self._get_producer_config()
            self.producer = KafkaProducer(**producer_config)
            
            print(f"[Producer] Starting to stream {video_path} to topic {self.topic_name}")
            
            # Open video file
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = video.get(cv2.CAP_PROP_FPS)
            
            # Use original FPS if not specified or interval not set
            if self.fps is None or self.interval is None or self.interval <= 0:
                self.fps = original_fps if original_fps > 0 else 12.0
                self.interval = 1.0 / self.fps
                print(f"[Producer] Using video FPS: {original_fps}, target FPS: {self.fps}, interval: {self.interval}")
            else:
                print(f"[Producer] Using specified FPS: {self.fps}, interval: {self.interval}")
            
            print(f"[Producer] Video info - Total frames: {total_frames}, Original FPS: {original_fps}, Target FPS: {self.fps}")
            
            # Create progress bar
            progress_bar = tqdm.tqdm(
                total=total_frames,
                desc=f"Streaming {os.path.basename(video_path)}",
                unit="frames"
            )
            
            frame_count = 0
            last_time = time.time()
            
            while self.is_streaming and video.isOpened():
                success, frame = video.read()
                if not success:
                    print(f"[Producer] End of video reached for {self.topic_name}")
                    break
                
                # Process and send frame
                self._encode_and_produce(frame, frame_count)
                frame_count += 1
                progress_bar.update(1)
                
                # Frame rate control
                if self.interval and self.interval > 0:
                    current_time = time.time()
                    elapsed = current_time - last_time
                    sleep_time = self.interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    last_time = time.time()
            
            progress_bar.close()
            video.release()
            
            # Send end-of-stream marker
            if self.is_streaming:
                print(f"[Producer] Completed streaming all frames for {self.topic_name}")
                self._send_end_marker()
            
        except Exception as e:
            print(f"[Producer] Error in streaming: {e}")
        finally:
            self._cleanup()
    
    def publish_video(self, source: str):
        """
        Publish video frames to Kafka topic (batch mode).
        Compatible with Producer.py usage.
        
        Args:
            source (str): Path to video file or directory of images
        """
        try:
            # Initialize producer for batch mode
            producer_config = self._get_producer_config()
            self.producer = KafkaProducer(**producer_config)
            
            if os.path.isfile(source):
                print(f"[Producer] Publishing video {source} to topic {self.topic_name}")
                self._publish_from_video(source)
            elif os.path.isdir(source):
                print(f"[Producer] Publishing from image folder {source} to topic {self.topic_name}")
                self._publish_from_images(source)
            else:
                raise ValueError(f"Invalid source path: {source}")
            
            print('[Producer] Publishing complete!')
            
        except KeyboardInterrupt:
            print("[Producer] Publishing stopped by user.")
        except Exception as e:
            print(f"[Producer] Error in publishing: {e}")
        finally:
            self._cleanup()
    
    def _publish_from_video(self, video_path: str):
        """Publish frames from a video file."""
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = video.get(cv2.CAP_PROP_FPS)
        
        # Use original FPS if interval not specified or invalid
        if self.interval is None or self.interval <= 0:
            self.interval = 1.0 / original_fps if original_fps > 0 else 1.0 / 12
            print(f"[Producer] Using video FPS: {original_fps}, interval: {self.interval}")
        else:
            print(f"[Producer] Using specified interval: {self.interval}")
        
        progress_bar = tqdm.tqdm(
            total=total_frames,
            desc=f"Publishing {os.path.basename(video_path)}",
            unit="frames"
        )
        
        frame_count = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            
            # Only process every 5th frame for batch mode to reduce load
            if frame_count % 5 == 0:
                self._encode_and_produce(frame, frame_count, use_interval=True)
            
            frame_count += 1
            progress_bar.update(1)
        
        progress_bar.close()
        video.release()
    
    def _publish_from_images(self, image_dir: str):
        """Publish frames from a directory of images."""
        image_files = [f for f in os.listdir(image_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        # Default to 12 FPS for image sequences if interval not specified or invalid
        if self.interval is None or self.interval <= 0:
            self.interval = 1.0 / 12
            print(f"[Producer] Using default 12 FPS for images, interval: {self.interval}")
        else:
            print(f"[Producer] Using specified interval: {self.interval}")
        
        print(f"[Producer] Found {len(image_files)} images")
        
        for i, img_file in enumerate(sorted(image_files)):
            image_path = os.path.join(image_dir, img_file)
            frame = cv2.imread(image_path)
            if frame is not None:
                self._encode_and_produce(frame, i, use_interval=True)
            else:
                print(f"[Producer] Warning: Could not read image {img_file}")
    
    def _encode_and_produce(self, frame: np.ndarray, frame_count: int, use_interval: bool = False):
        """
        Process and send a single frame to Kafka.
        
        Args:
            frame: OpenCV frame
            frame_count: Frame number
            use_interval: Whether to apply interval delay
        """
        try:
            # Process frame
            processed_frame = self._process_frame(frame)
            
            # Convert frame to JPEG
            if self.mode == 'streaming':
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 80]
            else:
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
            
            success, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
            if not success:
                print(f"[Producer] Error: Could not encode frame {frame_count}")
                return
            
            # Send to Kafka
            key = f"frame_{frame_count}".encode('utf-8')
            self.producer.send(
                self.topic_name,
                key=key,
                value=buffer.tobytes()
            )
            
            # Clean up memory
            del processed_frame, buffer
            
            # Periodic garbage collection
            if frame_count % 100 == 0:
                gc.collect()
            
            # Apply interval delay if requested
            if use_interval and self.interval and self.interval > 0:
                time.sleep(self.interval)
                
        except Exception as e:
            print(f"[Producer] Error encoding frame {frame_count}: {e}")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame before sending.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        # Resize while maintaining aspect ratio
        target_width = 1280  # Standard HD width
        height, width = frame.shape[:2]
        
        if width > target_width:
            aspect_ratio = width / height
            target_height = int(target_width / aspect_ratio)
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def _send_end_marker(self):
        """Send end-of-stream marker."""
        try:
            self.producer.send(
                self.topic_name,
                key=b"END_OF_STREAM",
                value=b"END_OF_STREAM"
            )
            self.producer.flush()
            print(f"[Producer] End-of-stream marker sent for {self.topic_name}")
            
            # Wait for processing to complete
            time.sleep(2)
            
        except Exception as e:
            print(f"[Producer] Error sending end-of-stream marker: {e}")
    
    def stop_streaming(self):
        """Stop streaming."""
        self.is_streaming = False
        print(f"[Producer] Stopping stream to {self.topic_name}")
    
    def _cleanup(self):
        """Clean up resources."""
        if self.producer:
            try:
                self.producer.flush()
                self.producer.close()
                print(f"[Producer] Producer closed for {self.topic_name}")
            except Exception as e:
                print(f"[Producer] Error during cleanup: {e}")
        
        # Force garbage collection
        gc.collect()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self._cleanup()


# Backward compatibility aliases
class VideoProducer_Legacy:
    """Legacy interface for backward compatibility with existing code."""
    
    def __init__(self, topic: str, interval: Union[float, int], bootstrap_servers: str = 'localhost:9092'):
        self.producer = VideoProducer(
            topic=topic,
            bootstrap_servers=bootstrap_servers,
            interval=interval,
            mode='batch'
        )
    
    def publish_video(self, source: str):
        return self.producer.publish_video(source)
    
    def encode_and_produce(self, frame, interval: float, frame_count: int = 0):
        # Legacy method - only process every 5th frame
        if frame_count % 5 == 0:
            self.producer._encode_and_produce(frame, frame_count, use_interval=True)


# Factory function for easy instantiation
def create_video_producer(topic: str, bootstrap_servers: str = 'localhost:9092', **kwargs):
    """
    Factory function to create VideoProducer with flexible parameters.
    
    Args:
        topic: Kafka topic name
        bootstrap_servers: Kafka bootstrap servers
        **kwargs: Additional parameters (fps, interval, mode, etc.)
    
    Returns:
        VideoProducer instance
    """
    return VideoProducer(topic=topic, bootstrap_servers=bootstrap_servers, **kwargs)
