import os
# os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
# os.environ["SPARK_HOME"] = r"C:\spark\spark-3.5.5-bin-hadoop3"
# os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["SPARK_HOME"] + r"\bin;" + os.environ["PATH"]

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

def process_messages(consumer: KafkaConsumer):
    global frame_counter
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

def save_processed_videos(processed_images, save_dir):
    """
    Save processed frames from the queue to video files, one per camera.
    Args:
        processed_images: Queue containing (consumer_name, frame) tuples.
        save_dir: Directory to save videos.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writers = {}
    frame_sizes = {}
    print("Saving processed videos. Press Ctrl+C to stop and save.")
    try:
        while True:
            consumer_name, frame = processed_images.get()
            if frame is None:
                print(f"Received None image from {consumer_name}, skipping...")
                continue
            if consumer_name not in writers:
                h, w = frame.shape[:2]
                output_path = os.path.join(save_dir, f"{consumer_name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writers[consumer_name] = cv2.VideoWriter(output_path, fourcc, 6, (w, h))
                frame_sizes[consumer_name] = (w, h)
                print(f"Started writing video for {consumer_name} to {output_path}")
            # Resize if needed
            w, h = frame_sizes[consumer_name]
            if (frame.shape[1], frame.shape[0]) != (w, h):
                frame = cv2.resize(frame, (w, h))
            writers[consumer_name].write(frame)
            print(f"Processed frame from {consumer_name}, size: {frame.shape}")
    except KeyboardInterrupt:
        print("Interrupted. Finalizing video files...")
    finally:
        for writer in writers.values():
            writer.release()
        print("All videos saved.")

def display_images():
    """Display the processed images in the main thread"""
    while True:
        # Get the next processed image and display it
        consumer_name, final_img = processed_images.get()
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

def ensure_kafka_topic(topic_name, bootstrap_servers):
    """Check if a Kafka topic exists, and create it if not."""
    admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    existing_topics = admin_client.list_topics()
    if topic_name not in existing_topics:
        try:
            admin_client.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
            print(f"Created Kafka topic: {topic_name}")
            return KafkaConsumer(
                topic_name,
                bootstrap_servers=[bootstrap_servers],
            )
        except Exception as e:
            print(f"Error creating topic {topic_name}: {e}")
    else:
        print(f"Kafka topic already exists: {topic_name}")
        # Return a consumer for the topic
        return KafkaConsumer(
                topic_name,
                bootstrap_servers=[bootstrap_servers],
            )
    admin_client.close()

def main():
    admin_client = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)
    print("Current kafka topics: {}".format(admin_client.list_topics()))
    time.sleep(5)  # Wait for a moment to ensure the topics are listed correctly
    admin_client.close()

    # Start Spark streaming
    from streaming.spark_services.spark_streaming import start_spark
    spark_thread = threading.Thread(target=start_spark)
    try:
        print("Starting Spark streaming...")
        spark_thread.start()
        print("Spark streaming started.")
    except KeyboardInterrupt:
        print("Spark streaming interrupted by user.")
        
    # Ensure Kafka topics exist
    consumer_00 = ensure_kafka_topic(TOPIC_1 + "_processed", BOOTSTRAP_SERVERS)
    if TOPIC_2 and TOPIC_2 != "NULL":
        consumer_01 = ensure_kafka_topic(TOPIC_2 + "_processed", BOOTSTRAP_SERVERS)
    else:
        consumer_01 = None
    thread_0, thread_1 = start_threads(consumer_00, consumer_01)
    print("Threads started for processing messages.")

    if args['save_dir']:
        save_processed_videos(processed_images, args['save_dir'])
    else:
        display_images()

    # Wait for both threads to finish
    thread_0.join()
    if thread_1 is not None:
        thread_1.join()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
