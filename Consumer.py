import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["SPARK_HOME"] = r"C:\spark\spark-3.5.5-bin-hadoop3"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["SPARK_HOME"] + r"\bin;" + os.environ["PATH"]

import argparse
import cv2
from queue import Queue
import numpy as np
import threading
from kafka import KafkaConsumer

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

def process_messages(consumer: KafkaConsumer,
                     consumer_name: str):
    for msg in consumer:
        # Process the message - directly display processed frames from Spark
        final_img = np.frombuffer(msg.value, dtype=np.uint8)
        final_img = cv2.imdecode(final_img, cv2.IMREAD_COLOR)
        processed_images.put((consumer_name, final_img))

def start_threads(consumer_00: KafkaConsumer,
                  consumer_01: KafkaConsumer):
    """Start processing messages from both topics using threads"""
    thread_0 = threading.Thread(
        target=process_messages,
        args=(consumer_00, "Camera 00")
    )
    thread_1 = threading.Thread(
        target=process_messages,
        args=(consumer_01, "Camera 01")
    )

    thread_0.start()
    thread_1.start()

    return thread_0, thread_1

def display_images():
    """Display the processed images in the main thread"""
    while True:
        # Get the next processed image and display it
        consumer_name, final_img = processed_images.get()
        cv2.imshow(consumer_name, final_img)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

def main():
    # Start Spark streaming
    from streaming.spark_services.spark_streaming import start_spark
    spark_thread = threading.Thread(target=start_spark)
    print("Starting Spark streaming...")
    spark_thread.start()
    print("Spark streaming started.")

    # Create Kafka consumers for processed topics
    consumer_00 = KafkaConsumer(
        TOPIC_1 + "_processed",
        bootstrap_servers=[BOOTSTRAP_SERVERS]
    )

    consumer_01 = KafkaConsumer(
        TOPIC_2 + "_processed",
        bootstrap_servers=[BOOTSTRAP_SERVERS]
    )
    print(f"Kafka consumers created for topics: {TOPIC_1}_processed, {TOPIC_2}_processed")
    
    thread_0, thread_1 = start_threads(consumer_00, consumer_01)
    print("Threads started for processing messages.")
    display_images()

    # Wait for both threads to finish
    thread_0.join()
    thread_1.join()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
