import os
import time
import cv2
import threading
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import tqdm

class VideoProducer:
    def __init__(
            self,
            video_path: str,
            topic_name: str,
            bootstrap_servers: str = 'localhost:9092',
            fps: float = None):
        
        self.video_path = video_path
        self.topic_name = topic_name
        self.bootstrap_servers = bootstrap_servers
        self.fps = fps
        self.interval = 1.0 / fps if fps else 0.1  # Default to 10 FPS if not specified
        self.producer = None
        self.is_streaming = False
        
        # Create topic if it doesn't exist
        self._create_topic()

    def _create_topic(self):
        """Create Kafka topic if it doesn't exist."""
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                client_id='video_producer'
            )
            
            topic_list = [NewTopic(
                name=self.topic_name,
                num_partitions=1,
                replication_factor=1
            )]
            
            admin_client.create_topics(new_topics=topic_list, validate_only=False)
            print(f"[Producer] Created topic: {self.topic_name}")
            
        except Exception as e:
            # Topic might already exist
            print(f"[Producer] Topic {self.topic_name} already exists")

    def start_streaming(self):
        """Start streaming video frames to Kafka."""
        if self.is_streaming:
            print(f"[Producer] Already streaming to {self.topic_name}")
            return

        self.is_streaming = True
        
        try:
            # Initialize Kafka producer for real-time streaming
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                max_request_size=10485760,  # 10MB
                buffer_memory=67108864,     # 64MB
                max_in_flight_requests_per_connection=5,  # Reduced from 1
                batch_size=16384,           # Increased batch size (16KB)
                linger_ms=10,               # Small batching delay (10ms)
                acks=1,                     # Wait for one broker to acknowledge
                compression_type='gzip',  # Add compression
                value_serializer=None,
                key_serializer=None,
            )
            
            print(f"[Producer] Starting to stream {self.video_path} to topic {self.topic_name}")
            
            # Open video file
            video = cv2.VideoCapture(self.video_path)
            if not video.isOpened():
                raise ValueError(f"Cannot open video file: {self.video_path}")
            
            # Get video properties
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = video.get(cv2.CAP_PROP_FPS)
            
            print(f"[Producer] Video info - Total frames: {total_frames}, FPS: {original_fps}")
            
            # Create progress bar
            progress_bar = tqdm.tqdm(
                total=total_frames,
                desc=f"Streaming {os.path.basename(self.video_path)}",
                unit="frames"
            )
            
            frame_count = 0
            
            while self.is_streaming and video.isOpened():
                success, frame = video.read()
                if not success:
                    print(f"[Producer] End of video reached for {self.topic_name}")
                    break
                
                # Process and send frame
                self._encode_and_produce(frame, frame_count)
                frame_count += 1
                progress_bar.update(1)
                
                if self.fps:
                    # Control frame rate
                    time.sleep(self.interval)
                
            progress_bar.close()
            video.release()
            
            # Signal completion but wait before exiting to allow processing to complete
            if self.is_streaming:
                print(f"[Producer] Completed streaming all frames for {self.topic_name}")
                
                # Send a final marker message to indicate end of stream
                try:
                    self.producer.send(
                        self.topic_name,
                        key=b"END_OF_STREAM",
                        value=b"END_OF_STREAM"
                    )
                    self.producer.flush()
                    print(f"[Producer] End-of-stream marker sent for {self.topic_name}")
                    
                    # Wait a bit to ensure all messages are processed
                    time.sleep(10)
                    print(f"[Producer] Stream {self.topic_name} fully completed")
                except Exception as e:
                    print(f"[Producer] Error sending end-of-stream marker: {e}")
            
        except Exception as e:
            print(f"[Producer] Error in streaming: {e}")
        finally:
            if self.producer:
                self.producer.flush()
                self.producer.close()
            print(f"[Producer] Stopped streaming to {self.topic_name}")

    def _encode_and_produce(self, frame, frame_count: int):
        """Process and send a single frame to Kafka."""
        try:
            # Resize frame for better performance
            processed_frame = self._process_frame(frame)
            
            # Convert frame to jpg format
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 80]
            _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
            
            # Send to Kafka with frame count as key
            key = f"frame_{frame_count}".encode('utf-8')
            self.producer.send(
                self.topic_name, 
                key=key,
                value=buffer.tobytes()
            )
            
            # Clean up memory
            del processed_frame, buffer
            
            # Garbage collection less frequently for smoother streaming
            if frame_count % 1000 == 0:
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"[Producer] Error encoding frame {frame_count}: {e}")

    def _process_frame(self, frame):
        """Preprocess frame before sending."""
        # Resize while maintaining aspect ratio
        target_width = 1280  # Standard HD width
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        target_height = int(target_width / aspect_ratio)
        
        # Resize frame
        frame = cv2.resize(frame, (target_width, target_height))
        
        return frame

    def stop_streaming(self):
        """Stop streaming."""
        self.is_streaming = False
        print(f"[Producer] Stopping stream to {self.topic_name}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.producer:
            self.producer.close()
