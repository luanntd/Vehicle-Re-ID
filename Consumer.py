import os
import argparse
import cv2
from queue import Queue, Empty
import numpy as np
import threading
import time
from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic

DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"

def parse_args():
    """Parse User's input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--bootstrap-servers",
                        type=str,
                        default="localhost:9092",
                        help="The address of the Kafka bootstrap servers")
    parser.add_argument("-t", "--topic", "--topic-1",
                        type=str,
                        required=True,
                        help="The name of the first kafka topic")
    parser.add_argument("-t2", "--topic-2",
                        type=str,
                        default="NULL",
                        help="The name of the second kafka topic (optional)")
    parser.add_argument("--view",
                        action="store_true",
                        help="Show live OpenCV windows while consuming")
    parser.add_argument("--display-fps",
                        type=float,
                        default=6,
                        help="Refresh rate for preview windows (higher looks smoother)")
    parser.add_argument("--target-fps",
                        type=float,
                        default=6.0,
                        help="Max preview update rate per topic; set 0 to disable frame pacing")
    parser.add_argument("-s", "--save-dir",
                        type=str,
                        default="Vehicle-Re-ID/output_videos",
                        help="The directory to save the detected vehicles")
    parser.add_argument("--no-save",
                        action="store_true",
                        help="Disable saving videos even if --save-dir is set")
    return parser.parse_args()

args = vars(parse_args())
BOOTSTRAP_SERVERS = args['bootstrap_servers']
TOPIC_1 = args['topic']
TOPIC_2 = args['topic_2']

VIEW_ENABLED = bool(args.get('view'))
SAVE_DIR = args.get('save_dir')
SAVE_ENABLED = (not bool(args.get('no_save'))) and bool(SAVE_DIR)
DISPLAY_FPS = float(args.get('display_fps') or 30.0)
TARGET_FPS = float(args.get('target_fps') or 0.0)

latest_frames_display = {}
latest_frames_lock = threading.Lock()
latest_frames_event = threading.Event()

_last_preview_push_time = {}

processed_images_save = Queue()
# Frame counter for memory management
frame_counter = 0
# Timeout monitoring
last_message_time = time.time()
timeout_seconds = 300
should_exit = False
# Frame rate tracking for each camera
camera_frame_times = {}
camera_frame_counts = {}


def _make_placeholder_frame(text: str, width: int = 640, height: int = 480) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img

def process_messages(consumer: KafkaConsumer):
    global frame_counter, last_message_time
    print(f"Starting to consume messages from topic: {consumer.subscription()}")
    try:
        empty_polls = 0
        while not should_exit:
            batch = consumer.poll(timeout_ms=1000, max_records=10)
            if not batch:
                empty_polls += 1
                if empty_polls % 10 == 0:
                    print(f"[Consumer] No messages yet on {list(consumer.subscription())}")
                continue

            empty_polls = 0

            batch_count = sum(len(msgs) for msgs in batch.values())
            if frame_counter % 100 == 0:
                print(f"[Consumer] Polled {batch_count} messages")

            for _tp, messages in batch.items():
                for msg in messages:
                    try:
                        last_message_time = time.time()

                        if VIEW_ENABLED and (not SAVE_ENABLED) and TARGET_FPS > 0:
                            now = time.time()
                            last_t = _last_preview_push_time.get(msg.topic, 0.0)
                            min_interval = 1.0 / TARGET_FPS
                            if (now - last_t) < min_interval:
                                continue
                            _last_preview_push_time[msg.topic] = now

                        frame_buffer = np.frombuffer(msg.value, dtype=np.uint8)
                        final_img = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)

                        if final_img is None:
                            print(f"Received None image from {msg.topic}, skipping...")
                            del frame_buffer
                            continue

                        if VIEW_ENABLED:
                            if TARGET_FPS > 0:
                                now = time.time()
                                last_t = _last_preview_push_time.get(msg.topic, 0.0)
                                min_interval = 1.0 / TARGET_FPS
                                if (now - last_t) >= min_interval:
                                    _last_preview_push_time[msg.topic] = now
                                    with latest_frames_lock:
                                        latest_frames_display[msg.topic] = final_img
                                    latest_frames_event.set()
                            else:
                                with latest_frames_lock:
                                    latest_frames_display[msg.topic] = final_img
                                latest_frames_event.set()
                        if SAVE_ENABLED:
                            processed_images_save.put((msg.topic, final_img))

                        if frame_counter % 100 == 0:
                            print(f"Received message from {msg.topic}, frame size: {final_img.shape}")

                        del frame_buffer
                        frame_counter += 1

                        if frame_counter % 50 == 0:
                            import gc
                            gc.collect()
                            print(f"Processed {frame_counter} frames, memory cleaned")

                    except Exception as e:
                        print(f"Error processing message from {msg.topic}: {e}")
                        import gc
                        gc.collect()
    except KeyboardInterrupt:
        print("Message processing interrupted by user")
    except Exception as e:
        print(f"Fatal error in message processing: {e}")
        import gc
        gc.collect()
    finally:
        print("Waiting Kafka consumer")

def start_threads(consumer_00: KafkaConsumer,
                  consumer_01: KafkaConsumer):
    thread_0 = threading.Thread(
        target=process_messages,
        args=(consumer_00,)
    )
    if consumer_01 is not None:
        thread_1 = threading.Thread(
            target=process_messages,
            args=(consumer_01,)
        )
    else:
        thread_1 = None

    thread_0.start()
    if thread_1 is not None:
        thread_1.start()

    return thread_0, thread_1

def monitor_timeout():
    global last_message_time, timeout_seconds
    while True:
        time.sleep(10)  # Check every 5 seconds
        current_time = time.time()
        time_since_last_message = current_time - last_message_time
        
        if time_since_last_message > timeout_seconds:
            print(f"[TIMEOUT] No messages received for {timeout_seconds} seconds. Signaling exit...")
            
            global should_exit
            should_exit = True
            return
        elif time_since_last_message > timeout_seconds * 0.7:
            print(f"[WARNING] No messages for {time_since_last_message:.1f} seconds (timeout in {timeout_seconds - time_since_last_message:.1f}s)")
            time.sleep(5) 

def calculate_camera_fps(camera_name):
    global camera_frame_times, camera_frame_counts
    
    output_fps = 6.0
    
    print(f"[FPS] Camera {camera_name}: Output video will be {output_fps} FPS")
    return output_fps

def save_processed_videos(processed_images, save_dir):
    global should_exit
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writers = {}
    frame_sizes = {}
    written_counts = {}
    try:
        while not should_exit:
            try:
                consumer_name, frame = processed_images.get(timeout=1)
            except Empty:
                continue  
                
            if frame is None:
                print(f"Received None image from {consumer_name}, skipping...")
                continue
            if consumer_name not in writers:
                h, w = frame.shape[:2]
                output_path = os.path.join(save_dir, f"{consumer_name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                
                dynamic_fps = calculate_camera_fps(consumer_name)
                
                writers[consumer_name] = cv2.VideoWriter(output_path, fourcc, dynamic_fps, (w, h))
                frame_sizes[consumer_name] = (w, h)
                written_counts[consumer_name] = 0
                print(f"Started writing video for {consumer_name} to {output_path} at {dynamic_fps:.2f} FPS")
       
            w, h = frame_sizes[consumer_name]
            if (frame.shape[1], frame.shape[0]) != (w, h):
                frame = cv2.resize(frame, (w, h))
            writers[consumer_name].write(frame)
            written_counts[consumer_name] += 1
            if written_counts[consumer_name] % 50 == 0:
                print(f"[Save] {consumer_name}: wrote {written_counts[consumer_name]} frames")
            
        print("[FINALIZING] Processing remaining frames before exit...")
        remaining_frames = 0
        while not processed_images.empty():
            try:
                consumer_name, frame = processed_images.get_nowait()
                if frame is not None:
                    if consumer_name in writers:
                        w, h = frame_sizes[consumer_name]
                        if (frame.shape[1], frame.shape[0]) != (w, h):
                            frame = cv2.resize(frame, (w, h))
                        writers[consumer_name].write(frame)
                        remaining_frames += 1
            except:
                break
        
        if remaining_frames > 0:
            print(f"[FINALIZING] Processed {remaining_frames} remaining frames")
            
    except KeyboardInterrupt:
        print("Interrupted. Finalizing video files...")
        remaining_frames = 0
        while not processed_images.empty():
            try:
                consumer_name, frame = processed_images.get_nowait()
                if frame is not None and consumer_name in writers:
                    w, h = frame_sizes[consumer_name]
                    if (frame.shape[1], frame.shape[0]) != (w, h):
                        frame = cv2.resize(frame, (w, h))
                    writers[consumer_name].write(frame)
                    remaining_frames += 1
            except:
                break
        if remaining_frames > 0:
            print(f"[FINALIZING] Processed {remaining_frames} remaining frames from interrupt")
            
    finally:
        print("[FINALIZING] Releasing video writers...")
        for camera_name, writer in writers.items():
            if writer is not None:
                writer.release()
                print(f"[FINALIZING] Released writer for {camera_name}")
        print("All videos saved and finalized.")
        
        if should_exit:
            print("[EXIT] Timeout reached. Exiting application...")
            os._exit(0)

def display_images():
    global should_exit

    topics = [TOPIC_1] + ([TOPIC_2] if TOPIC_2 and TOPIC_2 != "NULL" else [])
    try:
        for t in topics:
            cv2.namedWindow(t, cv2.WINDOW_NORMAL)
            cv2.imshow(t, _make_placeholder_frame(f"Waiting for frames on {t}..."))
        cv2.waitKey(1)
    except Exception as e:
        print(f"[Consumer] OpenCV GUI is not available (cv2.imshow failed): {e}")
        print("[Consumer] Install GUI-enabled OpenCV: pip install opencv-python (not opencv-python-headless)")
        return
    
    timeout_thread = threading.Thread(target=monitor_timeout)
    timeout_thread.daemon = True
    timeout_thread.start()
    
    refresh_hz = max(1.0, float(DISPLAY_FPS))
    refresh_interval_s = 1.0 / refresh_hz

    while not should_exit:
        latest_frames_event.wait(timeout=refresh_interval_s)
        latest_frames_event.clear()

        with latest_frames_lock:
            snapshot = dict(latest_frames_display)

        for consumer_name, final_img in snapshot.items():
            if final_img is None:
                continue
            if final_img.shape[1] != 640 or final_img.shape[0] != 480:
                preview = cv2.resize(final_img, (640, 480))
            else:
                preview = final_img
            cv2.imshow(consumer_name, preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            should_exit = True
            break
            
    if should_exit:
        print("[EXIT] Timeout reached. Closing display windows...")
        cv2.destroyAllWindows()
        os._exit(0)

def ensure_kafka_topic(session_id, bootstrap_servers):
    admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    existing_topics = admin_client.list_topics()
    topic_names = [f'cam1_{session_id}', f'cam2_{session_id}', f'cam1_processed_{session_id}', f'cam2_processed_{session_id}']
    
    for topic_name in topic_names:
        if topic_name not in existing_topics:
            try:
                admin_client.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
                print(f"Created Kafka topic: {topic_name}")
            except Exception as e:
                print(f"Error creating topic {topic_name}: {e}")
        else:
            print(f"Kafka topic already exists: {topic_name}")
    
    admin_client.close()

def ensure_kafka_topic_with_retry(topic_name, bootstrap_servers, max_retries=30, retry_delay=2):
    admin_client = None
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        
        for attempt in range(max_retries):
            try:
                existing_topics = admin_client.list_topics()
                
                if topic_name not in existing_topics:
                    if attempt == 0: 
                        try:
                            admin_client.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
                            print(f"Created Kafka topic: {topic_name}")
                        except Exception as e:
                            print(f"Note: Could not create topic {topic_name}: {e}")
                            print("Topic might be created by Spark streaming...")
                    
                    print(f"Waiting for topic {topic_name} to be available (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"Kafka topic found: {topic_name}")
                    break
                    
            except Exception as e:
                print(f"Error checking topics (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise e
        else:
            raise Exception(f"Topic {topic_name} was not available after {max_retries} attempts")
        
        for attempt in range(5):
            try:
                consumer = KafkaConsumer(
                    topic_name,
                    bootstrap_servers=[bootstrap_servers],
                    auto_offset_reset='latest', 
                    value_deserializer=None,     
                    fetch_min_bytes=1,
                    fetch_max_wait_ms=500,
                    max_poll_records=10,
                    enable_auto_commit=True,
                    auto_commit_interval_ms=1000,
                    fetch_max_bytes=52428800,      
                    max_in_flight_requests_per_connection=5,
                    receive_buffer_bytes=67108864,  
                    session_timeout_ms=30000,      
                    heartbeat_interval_ms=3000,    
                    max_poll_interval_ms=300000,    
                    group_id=f'consumer_{topic_name}_{int(time.time())}'
                )
                
                consumer.poll(timeout_ms=1000)
                print(f"Successfully connected to Kafka topic: {topic_name}")
                return consumer
                
            except Exception as e:
                print(f"Error connecting to consumer (attempt {attempt + 1}/5): {e}")
                if attempt < 4:
                    time.sleep(2)
                    continue
                else:
                    raise e
        
    except Exception as e:
        print(f"Failed to connect to Kafka topic {topic_name}: {e}")
        raise e
    finally:
        if admin_client:
            admin_client.close()

def verify_spark_output(topic_name, bootstrap_servers, timeout=60):
    print(f"Verifying Spark output to topic: {topic_name}")
    
    try:
        test_consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=[bootstrap_servers],
            auto_offset_reset='latest',
            value_deserializer=None,
            consumer_timeout_ms=timeout * 1000, 
            fetch_min_bytes=1,
            fetch_max_wait_ms=1000
        )
        
        print(f"Waiting up to {timeout} seconds for Spark to produce data...")
        
        messages = test_consumer.poll(timeout_ms=timeout * 1000)
        
        if messages:
            message_count = sum(len(msgs) for msgs in messages.values())
            print(f"✓ Spark is producing data to {topic_name} ({message_count} messages found)")
            test_consumer.close()
            return True
        else:
            print(f"✗ No data found in {topic_name} after {timeout} seconds")
            test_consumer.close()
            return False
            
    except Exception as e:
        print(f"Error verifying Spark output: {e}")
        return False

def main():
    processed_topic1 = TOPIC_1
    processed_topic2 = TOPIC_2 if TOPIC_2 and TOPIC_2 != "NULL" else None
    print(f"[Consumer] Consuming topics: {processed_topic1}, {processed_topic2}")
    consumer_00 = ensure_kafka_topic_with_retry(processed_topic1, BOOTSTRAP_SERVERS, max_retries=60, retry_delay=2)
    consumer_01 = ensure_kafka_topic_with_retry(processed_topic2, BOOTSTRAP_SERVERS, max_retries=60, retry_delay=2) if processed_topic2 else None
    
    thread_0, thread_1 = start_threads(consumer_00, consumer_01)
    print("Threads started for processing messages.")

    writer_thread = None

    try:
        if VIEW_ENABLED:
            if SAVE_ENABLED:
                writer_thread = threading.Thread(
                    target=save_processed_videos,
                    args=(processed_images_save, SAVE_DIR),
                    daemon=True,
                )
                writer_thread.start()
            display_images()
        else:
            if SAVE_ENABLED:
                save_processed_videos(processed_images_save, SAVE_DIR)
            else:
                display_images()
    except KeyboardInterrupt:
        print("Consumer interrupted by user.")
    finally:
        print("Shutting down consumer threads...")
        
        if thread_0.is_alive():
            thread_0.join(timeout=5)
        if thread_1 is not None and thread_1.is_alive():
            thread_1.join(timeout=5)

        if writer_thread is not None and writer_thread.is_alive():
            writer_thread.join(timeout=5)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
