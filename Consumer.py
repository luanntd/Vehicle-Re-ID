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
        print("Closing Kafka consumer")
        consumer.close()

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
    
    # Since producer sends every 5th frame from 30 FPS original
    # Output should be 30/5 = 6 FPS to maintain same duration
    output_fps = 6.0
    
    print(f"[FPS] Camera {camera_name}: Output video will be {output_fps} FPS (maintains original duration)")
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

def ensure_kafka_topic(topic_name, bootstrap_servers):
    """Check if a Kafka topic exists, and create it if not."""
    admin_client = None
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        existing_topics = admin_client.list_topics()
        
        if topic_name not in existing_topics:
            try:
                admin_client.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
                print(f"Created Kafka topic: {topic_name}")
                # Wait a bit for topic to be fully created
                time.sleep(2)
            except Exception as e:
                print(f"Error creating topic {topic_name}: {e}")
                # Topic might already exist due to race condition, continue anyway
        else:
            print(f"Kafka topic already exists: {topic_name}")
        
        # Return a consumer for the topic
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=[bootstrap_servers],
            auto_offset_reset='latest',  # Start from latest messages
            value_deserializer=None,     # We handle bytes manually
            consumer_timeout_ms=1000     # Timeout for testing if topic exists
        )
        
        print(f"Successfully connected to Kafka topic: {topic_name}")
        return consumer
        
    except Exception as e:
        print(f"Failed to connect to Kafka topic {topic_name}: {e}")
        raise e
    finally:
        if admin_client:
            admin_client.close()

def main():
    admin_client = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)
    print("Current kafka topics: {}".format(admin_client.list_topics()))
    time.sleep(5)  # Wait for a moment to ensure the topics are listed correctly
    admin_client.close()

    # Start Spark streaming with proper error handling
    from streaming.spark_services.spark_streaming import start_spark
    spark_thread = None
    try:
        print("Starting Spark streaming...")
        spark_thread = start_spark()  # start_spark() returns the thread
        print("Spark streaming started.")
        
        # Wait a bit for Spark to initialize and create output topics
        print("Waiting for Spark streaming to initialize...")
        time.sleep(10)
        
    except Exception as e:
        print(f"Error starting Spark streaming: {e}")
        print("Consumer will exit since Spark streaming failed to start.")
        return
        
    # Ensure Kafka topics exist (topics that Spark creates)
    processed_topic1 = TOPIC_1 + "_processed"
    processed_topic2 = TOPIC_2 + "_processed" if TOPIC_2 and TOPIC_2 != "NULL" else None
    
    try:
        consumer_00 = ensure_kafka_topic(processed_topic1, BOOTSTRAP_SERVERS)
        if processed_topic2:
            consumer_01 = ensure_kafka_topic(processed_topic2, BOOTSTRAP_SERVERS)
        else:
            consumer_01 = None
    except Exception as e:
        print(f"Error creating/connecting to Kafka topics: {e}")
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
