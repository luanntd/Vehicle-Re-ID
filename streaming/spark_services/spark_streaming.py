import threading
import numpy as np
import cv2
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType
from realtime_reid.pipeline import Pipeline
from realtime_reid.vehicle_detector import VehicleDetector
from realtime_reid.feature_extraction import VehicleDescriptor

def start_spark():
    # Use Spark/Scala/Kafka versions that actually exist and match your Spark install
    SCALA_VERSION = '2.13'  # Change to your Scala version if different
    SPARK_VERSION = '4.0.0'  # Change to your installed Spark version if different
    KAFKA_VERSION = '3.5.0'  # Should match Spark version, not Kafka server version

    packages = [
        f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
        f'org.apache.kafka:kafka-clients:{KAFKA_VERSION}'
    ]

    findspark.init()

    # Initialize the shared pipeline ONCE at startup (before Spark session)
    print("[Startup] Initializing shared pipeline for all cameras")
    detector = VehicleDetector(model_path='best_20.pt')
    descriptor = VehicleDescriptor(model_type='osnet')
    shared_pipeline = Pipeline(detector=detector, descriptor=descriptor)
    print("[Startup] Shared pipeline initialized successfully")

    def spark_streaming_thread():
        # Initialize Spark session first
        spark = SparkSession.builder \
            .master('local') \
            .appName("vehicle-reid") \
            .config("spark.jars.packages", ",".join(packages)) \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

        spark.conf.set("spark.sql.shuffle.partitions", "1")  # Use single partition for shared state

        # Define Kafka parameters
        kafka_params = {
            "kafka.bootstrap.servers": "localhost:9092",
            "subscribe": "cam1, cam2"
        }

        # Create streaming DataFrame with Kafka source
        df = spark.readStream \
            .format("kafka") \
            .options(**kafka_params) \
            .option("startingOffsets", "latest") \
            .load()

        df = df.withColumn("value", df["value"].cast(BinaryType()))

        @udf(BinaryType())
        def process_frame(value, topic):
            """Process individual frames using the shared pipeline."""
            print(f"[UDF] Processing frame from {topic} using shared pipeline")
            try:
                # Decode bytes to OpenCV frame
                frame_buffer = np.frombuffer(value, dtype=np.uint8)
                frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print(f"[UDF] ERROR: Could not decode frame from {topic}")
                    return value  # Return original bytes if decoding fails
                
                print(f"[UDF] Frame decoded successfully: {frame.shape} from {topic}")
                
                # Process frame with SHARED PIPELINE (same instance for all topics)
                processed_frame = shared_pipeline.process(frame, topic)
                
                # Encode result back to bytes for Kafka
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result_bytes = buffer.tobytes()
                
                print(f"[UDF] Processing completed for {topic}")
                
                # Clean up memory
                del frame_buffer, frame, processed_frame, buffer
                
                return result_bytes
                
            except Exception as e:
                print(f"[UDF] ERROR processing frame from {topic}: {e}")
                import gc
                gc.collect()
                return value  # Return original bytes on error

        # Process frames using UDF (now with shared pipeline from outer scope)
        processed_df = df \
            .selectExpr("CAST(key AS STRING) as key",
                       "CAST(topic as STRING) as topic",
                       "value") \
            .withColumn("value", process_frame("value", "topic"))

        # Define output Kafka parameters
        write_params = [
            {
                "kafka.bootstrap.servers": "localhost:9092",
                "topic": "cam1_processed"
            },
            {
                "kafka.bootstrap.servers": "localhost:9092", 
                "topic": "cam2_processed"
            }
        ]

        # Write processed frames back to Kafka
        query_topic1 = processed_df \
            .filter("topic = 'cam1'") \
            .select("key", "value") \
            .writeStream \
            .format("kafka") \
            .options(**write_params[0]) \
            .option("checkpointLocation", "tmp/cam1_processed") \
            .outputMode("append") \
            .start()

        query_topic2 = processed_df \
            .filter("topic = 'cam2'") \
            .select("key", "value") \
            .writeStream \
            .format("kafka") \
            .options(**write_params[1]) \
            .option("checkpointLocation", "tmp/cam2_processed") \
            .outputMode("append") \
            .start()

        query_topic1.awaitTermination()
        query_topic2.awaitTermination()

    thread = threading.Thread(target=spark_streaming_thread)
    thread.start()
    return thread
