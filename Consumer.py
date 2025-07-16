import os
import argparse
import cv2
from queue import Queue
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
    parser.add_argument("-s", "--save-dir",
                        type=str,
                        default=None,
                        help="The directory to save the detected vehicles")
    return parser.parse_args()

args = vars(parse_args())
BOOTSTRAP_SERVERS = args['bootstrap_servers']
TOPIC_1 = args['topic']
TOPIC_2 = args['topic_2']

# Create a Queue to hold the processed images
processed_images = Queue()
# Frame counter for memory management
frame_counter = 0
# Timeout monitoring
last_message_time = time.time()
timeout_seconds = 300
should_exit = False
# Frame rate tracking for each camera
camera_frame_times = {}
camera_frame_counts = {}

def process_messages(consumer: KafkaConsumer):
    global frame_counter
    print(f"Starting to consume messages from topic: {consumer.subscription()}")
    try:
        for msg in consumer:
            try:
                # Process the message - directly display processed frames from Spark
                frame_buffer = np.frombuffer(msg.value, dtype=np.uint8)
                final_img = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                
                # Check if final_img is None
                if final_img is None:
                    print(f"Received None image from {msg.topic}, skipping...")
                    del frame_buffer  # Clean up even on failure
                    continue
                    
                processed_images.put((msg.topic, final_img))
                print(f"Received message from {msg.topic}, frame size: {final_img.shape}")
                
                # Clean up memory
                del frame_buffer
                frame_counter += 1
                
                # Garbage collection every 50 frames
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
        # consumer.close()

def start_threads(consumer_00: KafkaConsumer,
                  consumer_01: KafkaConsumer):
    """Start processing messages from both topics using threads"""
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
    """Monitor for timeout and exit if no messages received for timeout_seconds"""
    global last_message_time, timeout_seconds
    while True:
        time.sleep(10)  # Check every 5 seconds
        current_time = time.time()
        time_since_last_message = current_time - last_message_time
        
        if time_since_last_message > timeout_seconds:
            print(f"[TIMEOUT] No messages received for {timeout_seconds} seconds. Signaling exit...")
            # Set a global flag instead of force exit to allow cleanup
            global should_exit
            should_exit = True
            return
        elif time_since_last_message > timeout_seconds * 0.7:  # Warning at 70% of timeout
            print(f"[WARNING] No messages for {time_since_last_message:.1f} seconds (timeout in {timeout_seconds - time_since_last_message:.1f}s)")
            time.sleep(5)  # Sleep to avoid spamming warnings

def calculate_camera_fps(camera_name):
    """Calculate the output FPS to maintain same duration as original video"""
    global camera_frame_times, camera_frame_counts
    
    # Fixed 6 FPS output as requested
    output_fps = 6.0
    
    print(f"[FPS] Camera {camera_name}: Output video will be {output_fps} FPS")
    return output_fps

def save_processed_videos(processed_images, save_dir):
    """
    Save processed frames from the queue to video files, one per camera.
    Args:
        processed_images: Queue containing (consumer_name, frame) tuples.
        save_dir: Directory to save videos.
    """
    global should_exit
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writers = {}
    frame_sizes = {}
    print("Saving processed videos. Press Ctrl+C to stop and save.")
    
    # # Start timeout monitor thread
    # timeout_thread = threading.Thread(target=monitor_timeout)
    # timeout_thread.daemon = True
    # timeout_thread.start()
    
    try:
        while not should_exit:
            try:
                consumer_name, frame = processed_images.get()  # 1 second timeout
            except:
                continue  # Continue checking for timeout and should_exit flag
                
            if frame is None:
                print(f"Received None image from {consumer_name}, skipping...")
                continue
            if consumer_name not in writers:
                h, w = frame.shape[:2]
                output_path = os.path.join(save_dir, f"{consumer_name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                # Calculate dynamic FPS based on actual processing rate
                dynamic_fps = calculate_camera_fps(consumer_name)
                
                writers[consumer_name] = cv2.VideoWriter(output_path, fourcc, dynamic_fps, (w, h))
                frame_sizes[consumer_name] = (w, h)
                print(f"Started writing video for {consumer_name} to {output_path} at {dynamic_fps:.2f} FPS")
            # Resize if needed
            w, h = frame_sizes[consumer_name]
            if (frame.shape[1], frame.shape[0]) != (w, h):
                frame = cv2.resize(frame, (w, h))
            writers[consumer_name].write(frame)
            print(f"Processed frame from {consumer_name}, size: {frame.shape}")
            
        # Process any remaining frames in queue before finalizing
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
        # Process any remaining frames even on interrupt
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
        # Properly release all video writers
        print("[FINALIZING] Releasing video writers...")
        for camera_name, writer in writers.items():
            if writer is not None:
                writer.release()
                print(f"[FINALIZING] Released writer for {camera_name}")
        print("All videos saved and finalized.")
        
        # Force exit after cleanup
        if should_exit:
            print("[EXIT] Timeout reached. Exiting application...")
            os._exit(0)

def display_images():
    """Display the processed images in the main thread"""
    global should_exit
    
    # Start timeout monitor thread
    timeout_thread = threading.Thread(target=monitor_timeout)
    timeout_thread.daemon = True
    timeout_thread.start()
    
    while not should_exit:
        try:
            # Get the next processed image and display it
            consumer_name, final_img = processed_images.get(timeout=1)  # 1 second timeout
        except:
            continue  # Continue checking for timeout and should_exit flag
            
        if final_img is None:
            print(f"Received None image from {consumer_name}, skipping...")
            continue
        final_img = cv2.resize(final_img, (640, 480))  # Resize for better visibility
        cv2.imshow(consumer_name, final_img)
        sleep_time = 0.5  # Adjust sleep time as needed
        time.sleep(sleep_time)
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    if should_exit:
        print("[EXIT] Timeout reached. Closing display windows...")
        cv2.destroyAllWindows()
        os._exit(0)

def ensure_kafka_topic(session_id, bootstrap_servers):
    """Check if a Kafka topic exists, and create it if not. Based on app.py approach."""
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
    """Enhanced topic verification with retry logic like app.py consumer approach."""
    admin_client = None
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        
        # Check if topic already exists
        for attempt in range(max_retries):
            try:
                existing_topics = admin_client.list_topics()
                
                if topic_name not in existing_topics:
                    if attempt == 0:  # Only try to create on first attempt
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
            # If we exit the loop without breaking, topic was not found
            raise Exception(f"Topic {topic_name} was not available after {max_retries} attempts")
        
        # Try to create consumer with retries (similar to app.py kafka_consumer_thread)
        for attempt in range(5):
            try:
                consumer = KafkaConsumer(
                    topic_name,
                    bootstrap_servers=[bootstrap_servers],
                    auto_offset_reset='latest',  # Start from latest messages
                    value_deserializer=None,     # We handle bytes manually
                    consumer_timeout_ms=2000,    # Increased timeout like app.py
                    fetch_min_bytes=1,
                    fetch_max_wait_ms=500,
                    max_poll_records=10,
                    enable_auto_commit=True,
                    auto_commit_interval_ms=1000,
                    fetch_max_bytes=52428800,       # 50MB like app.py
                    max_in_flight_requests_per_connection=5,
                    receive_buffer_bytes=67108864,  # 64MB like app.py
                    session_timeout_ms=30000,       # Increased session timeout
                    heartbeat_interval_ms=3000,     # Heartbeat every 3 seconds
                    max_poll_interval_ms=300000,    # 5 minutes max poll interval
                    group_id=f'consumer_{topic_name}_{int(time.time())}'  # Unique group ID
                )
                
                # Test the connection
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
    """Verify that Spark is actually producing data to the processed topic."""
    print(f"Verifying Spark output to topic: {topic_name}")
    
    try:
        # Create a temporary consumer to check for messages
        test_consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=[bootstrap_servers],
            auto_offset_reset='latest',
            value_deserializer=None,
            consumer_timeout_ms=timeout * 1000,  # Convert to ms
            fetch_min_bytes=1,
            fetch_max_wait_ms=1000
        )
        
        print(f"Waiting up to {timeout} seconds for Spark to produce data...")
        
        # Try to get at least one message
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

def check_chromadb_setup():
    """Check if ChromaDB is properly set up."""
    try:
        # Try to import ChromaDB
        import chromadb
        print("✓ ChromaDB is installed and importable")
        
        # Try to create a test client
        test_client = chromadb.Client()
        print("✓ ChromaDB client can be created")
        
        return True
        
    except ImportError as e:
        print(f"✗ ChromaDB import error: {e}")
        print("Please install ChromaDB: pip install chromadb")
        return False
    except Exception as e:
        print(f"✗ ChromaDB setup error: {e}")
        return False

def main():
    # Extract session ID from topics (assuming format: cam1_sessionId, cam2_sessionId)
    session_id = None
    if "_" in TOPIC_1:
        session_id = TOPIC_1.split("_", 1)[1]
    elif "_" in TOPIC_2 and TOPIC_2 != "NULL":
        session_id = TOPIC_2.split("_", 1)[1]
    
    if not session_id:
        print("ERROR: Could not extract session ID from topics. Please use format: cam1_sessionId, cam2_sessionId")
        return
    
    print(f"Extracted session ID: {session_id}")
    
    # Use app.py approach for topic management
    print("Creating Kafka topics using app.py approach...")
    ensure_kafka_topic(session_id, BOOTSTRAP_SERVERS)
    
    admin_client = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)
    current_topics = admin_client.list_topics()
    print("Current kafka topics: {}".format(current_topics))
    
    # Check if input topics exist
    if TOPIC_1 not in current_topics:
        print(f"ERROR: Input topic '{TOPIC_1}' does not exist!")
        print("Please create the input topic first or run the producer.")
        admin_client.close()
        return
        
    if TOPIC_2 and TOPIC_2 != "NULL" and TOPIC_2 not in current_topics:
        print(f"ERROR: Input topic '{TOPIC_2}' does not exist!")
        print("Please create the input topic first or run the producer.")
        admin_client.close()
        return
        
    print("✓ Input topics verified.")
    admin_client.close()

    # Check ChromaDB setup
    print("Checking ChromaDB setup...")
    if not check_chromadb_setup():
        print("ChromaDB setup failed. Exiting.")
        return

    # Start Spark streaming with proper error handling
    from streaming.spark_streaming import start_spark
    spark_thread = None
    try:
        print("Starting Spark streaming...")
        # Use app.py approach - pass session_id and base_dir
        from pathlib import Path
        base_dir = Path(__file__).resolve().parent
        spark_thread = start_spark(session_id, base_dir=base_dir)
        print("Spark streaming started.")
        
        # Wait longer for Spark to initialize and create output topics (like app.py)
        print("Waiting for Spark streaming to initialize and create output topics...")
        time.sleep(30)  # Same as app.py
        
    except Exception as e:
        print(f"Error starting Spark streaming: {e}")
        print("Consumer will exit since Spark streaming failed to start.")
        return
        
    # Generate processed topic names using session-based approach from app.py
    processed_topic1 = f"{TOPIC_1}_processed"
    processed_topic2 = f"{TOPIC_2}_processed" if TOPIC_2 and TOPIC_2 != "NULL" else None
    
    print(f"Looking for processed topics: {processed_topic1}, {processed_topic2}")
    
    try:
        # Wait for Spark to create the processed topics (using app.py approach)
        print("Connecting to processed topics...")
        consumer_00 = ensure_kafka_topic_with_retry(processed_topic1, BOOTSTRAP_SERVERS, max_retries=60, retry_delay=5)
        
        if processed_topic2:
            consumer_01 = ensure_kafka_topic_with_retry(processed_topic2, BOOTSTRAP_SERVERS, max_retries=60, retry_delay=5)
        else:
            consumer_01 = None
            
        print("Successfully connected to all required topics.")
        
        # Verify that Spark is actually producing data
        print("Verifying Spark streaming is working...")
        if not verify_spark_output(processed_topic1, BOOTSTRAP_SERVERS, timeout=60):
            print("WARNING: Spark doesn't seem to be producing data to processed topics.")
            print("This could mean:")
            print("1. Input topics are empty")
            print("2. Spark streaming encountered an error")
            print("3. ChromaDB configuration issues")
            print("Continuing anyway...")
        
    except Exception as e:
        print(f"Error creating/connecting to Kafka topics: {e}")
        print("This usually means Spark streaming hasn't started producing to processed topics yet.")
        print("Make sure:")
        print("1. Kafka is running")
        print("2. Input topics exist and have data")
        print("3. Spark streaming is processing correctly")
        return
        
    # Verify Spark is producing data to the processed topics (using enhanced verification)
    try:
        if not verify_spark_output(processed_topic1, BOOTSTRAP_SERVERS, timeout=60):
            print(f"Spark does not appear to be producing data to {processed_topic1}.")
            print("This might be normal if input topics are empty or Spark is still initializing.")
            print("Continuing with consumer setup...")
        
        if processed_topic2 and processed_topic2 != "NULL":
            if not verify_spark_output(processed_topic2, BOOTSTRAP_SERVERS, timeout=60):
                print(f"Spark does not appear to be producing data to {processed_topic2}.")
                print("This might be normal if input topics are empty or Spark is still initializing.")
                print("Continuing with consumer setup...")
    except Exception as e:
        print(f"Error verifying Spark output: {e}")
        print("Continuing with consumer setup...")
    
    # ChromaDB setup verification
    try:
        print("Final ChromaDB setup verification...")
        if not check_chromadb_setup():
            print("ChromaDB setup is not correct. Please check the installation and configuration.")
            return
    except Exception as e:
        print(f"Error checking ChromaDB setup: {e}")
        return
    
    thread_0, thread_1 = start_threads(consumer_00, consumer_01)
    print("Threads started for processing messages.")

    try:
        if args['save_dir']:
            save_processed_videos(processed_images, args['save_dir'])
        else:
            display_images()
    except KeyboardInterrupt:
        print("Consumer interrupted by user.")
    finally:
        # Clean shutdown
        print("Shutting down consumer threads...")
        # Note: In a production system, you'd want to implement proper thread shutdown
        
        # Wait for both threads to finish
        if thread_0.is_alive():
            thread_0.join(timeout=5)
        if thread_1 is not None and thread_1.is_alive():
            thread_1.join(timeout=5)

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
