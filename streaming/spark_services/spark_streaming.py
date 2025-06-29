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

    def spark_streaming_thread():
        detector = VehicleDetector(model_path='best_20.pt')
        descriptor = VehicleDescriptor(model_type='osnet')
        vehicle_pipeline= Pipeline(detector=detector, descriptor=descriptor)

        # Initialize Spark session
        spark = SparkSession.builder \
            .master('local') \
            .appName("vehicle-reid") \
            .config("spark.jars.packages", ",".join(packages)) \
            .getOrCreate()

        spark.conf.set("spark.sql.shuffle.partitions", "20")

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
        def process_frame(value, tag):
            print("[UDF] process_frame called")
            try:
                # Decode bytes to OpenCV frame
                frame_buffer = np.frombuffer(value, dtype=np.uint8)
                frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("[UDF] ERROR: Could not decode frame")
                    return value  # Return original bytes if decoding fails
                
                print(f"[UDF] Frame decoded successfully: {frame.shape}")
                
                # Process frame directly (no encoding/decoding)
                processed_frame = vehicle_pipeline.process(frame, tag)
                
                # Encode result back to bytes for Kafka
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result_bytes = buffer.tobytes()
                
                print("[UDF] vehicle_pipeline.process finished")
                
                # Clean up memory
                del frame_buffer, frame, processed_frame, buffer
                
                return result_bytes
                
            except Exception as e:
                print(f"[UDF] ERROR in process_frame: {e}")
                import gc
                gc.collect()
                return value  # Return original bytes on error

        # Process frames
        processed_df = df \
            .selectExpr("CAST(key AS STRING)",
                       "CAST(topic as STRING)",
                       "value") \
            .withColumn("value", process_frame("value", "topic"))

        # Define output Kafka parameters
        write_params = [
            {
                "kafka.bootstrap.servers": "localhost:9092",
                "topic": kafka_params["subscribe"].split(", ")[i] + "_processed"
            } for i in range(2)
        ]

        # Write processed frames back to Kafka
        query_topic1 = processed_df \
            .filter("topic = 'cam1'") \
            .writeStream \
            .format("kafka") \
            .options(**write_params[0]) \
            .option("checkpointLocation", "tmp/" + write_params[0]["topic"]) \
            .outputMode("append") \
            .start()

        query_topic2 = processed_df \
            .filter("topic = 'cam2'") \
            .writeStream \
            .format("kafka") \
            .options(**write_params[1]) \
            .option("checkpointLocation", "tmp/" + write_params[1]["topic"]) \
            .outputMode("append") \
            .start()

        query_topic1.awaitTermination()
        query_topic2.awaitTermination()

    thread = threading.Thread(target=spark_streaming_thread)
    thread.start()
    return thread
