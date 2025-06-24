import argparse
from streaming.kafka_services.video_producer import VideoProducer

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
    
    producer = VideoProducer(
        topic=args['topic'],
        interval=args['interval']
    )
    producer.publish_video(args['camera'])

if __name__ == '__main__':
    main()
