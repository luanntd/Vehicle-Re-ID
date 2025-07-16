import argparse
from streaming.kafka_producer import VideoProducer
from kafka.admin import KafkaAdminClient, NewTopic

# Default Kafka bootstrap servers
BOOTSTRAP_SERVERS = "localhost:9092"

def parse_args():
    parser = argparse.ArgumentParser(description='Publish video stream to Kafka')
    parser.add_argument("-t", "--topic",
                        type=str,
                        required=True,
                        help="Kafka topic name")
    parser.add_argument("-c", "--camera",
                        type=str,
                        required=True,
                        help="Path to video file or image directory")
    parser.add_argument("-i", "--interval",
                        type=float,
                        default=-1,
                        help="Frame interval in seconds or FPS if > 1. "
                        "Defaults to video FPS or 12 FPS for images")
    return parser.parse_args()

def main():
    """Publish video stream to Kafka topic."""
    args = vars(parse_args())
    
    admin_client = KafkaAdminClient(bootstrap_servers= BOOTSTRAP_SERVERS)
    topic = args['topic']
    existing_topics = admin_client.list_topics()
    if topic not in existing_topics:
        try:
            new_topic = NewTopic(name=topic, num_partitions=1, replication_factor=1)
            admin_client.create_topics([new_topic])
            print(f"Created new topic: {topic}")
        except Exception as e:
            print(f"Error creating topic {topic}: {e}")
            raise
    else:
        print(f"Using existing topic: {topic}")
    admin_client.close()
    
    # Create producer with batch mode for command-line usage
    producer = VideoProducer(
        topic=args['topic'],
        bootstrap_servers=BOOTSTRAP_SERVERS,
        interval=args['interval'],
        mode='batch'
    )
    producer.publish_video(args['camera'])

if __name__ == '__main__':
    main()
