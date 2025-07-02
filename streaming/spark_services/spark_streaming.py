import threading
import numpy as np
import cv2
import findspark
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType
from realtime_reid.pipeline import Pipeline
from realtime_reid.vehicle_detector import VehicleDetector
from realtime_reid.feature_extraction import VehicleDescriptor
from realtime_reid.classifier_chromadb import ChromaDBVehicleReID

# Global variable to hold the single shared pipeline instance with ChromaDB
# This will be initialized lazily to avoid serialization
_shared_pipeline = None

def get_or_create_shared_pipeline():
    """Get or create the shared pipeline instance with ChromaDB backend."""
    global _shared_pipeline
    if _shared_pipeline is None:
        print("[Pipeline Init] Creating CHROMADB-INTEGRATED PIPELINE for persistent cross-camera ReID")
        detector = VehicleDetector(model_path='best_20.pt')
        descriptor = VehicleDescriptor(model_type='osnet', model_path='best_osnet_model.pth')
        # Initialize ChromaDB classifier with persistent storage
        chromadb_classifier = ChromaDBVehicleReID(
            db_path="./chroma_vehicle_reid_streaming",
            collection_name="vehicle_embeddings_streaming"
        )
        # Create pipeline with ChromaDB classifier
        _shared_pipeline = Pipeline(
            detector=detector, 
            descriptor=descriptor, 
            classifier=chromadb_classifier
        )
        print("[Pipeline Init] CHROMADB-INTEGRATED PIPELINE created successfully")
        
        # Show initial database statistics
        print("[Pipeline Init] Initial ChromaDB statistics:")
        stats = chromadb_classifier.get_statistics()
        for vehicle_type, count in stats.items():
            if vehicle_type not in ['total', 'max_ids']:
                print(f"  {vehicle_type}: {count} embeddings")
        print(f"  Total: {stats['total']} embeddings")
        print(f"  Max IDs: {stats['max_ids']}")
        
    return _shared_pipeline

def start_spark():
    """
    Start Spark Streaming with CHROMADB-BACKED vehicle re-identification using UDF processing.
    
    Key Features:
    - Persistent vehicle tracking using ChromaDB vector database
    - Cross-camera vehicle re-identification with feature matching
    - Vehicle ID persistence across sessions and time periods
    - Uses UDF for frame-by-frame processing
    - ChromaDB-based similarity search for vehicle matching
    - Automatic cross-camera image saving
    """
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
        # Initialize Spark session first
        spark = SparkSession.builder \
            .master('local') \
            .appName("vehicle-reid-chromadb") \
            .config("spark.jars.packages", ",".join(packages)) \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.serializer.objectStreamReset", "1") \
            .config("spark.rdd.compress", "false") \
            .config("spark.kryo.unsafe", "true") \
            .config("spark.kryoserializer.buffer.max", "512m") \
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

        # UDF for processing frames with ChromaDB-backed pipeline
        @udf(BinaryType())
        def process_frame_with_chromadb(value, topic):
            """Process individual frames using ChromaDB-backed vehicle ReID pipeline."""
            print(f"[UDF-ChromaDB] Processing frame from {topic} using CHROMADB-BACKED PIPELINE")
            
            try:
                # Get the shared ChromaDB-backed pipeline instance
                # This avoids serialization issues by creating pipeline in each executor
                pipeline = get_or_create_shared_pipeline()
                
                # Decode bytes to OpenCV frame
                frame_buffer = np.frombuffer(value, dtype=np.uint8)
                frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print(f"[UDF-ChromaDB] ERROR: Could not decode frame from {topic}")
                    return value  # Return original bytes if decoding fails
                
                print(f"[UDF-ChromaDB] Frame decoded successfully: {frame.shape} from {topic}")
                
                # Process frame with CHROMADB-BACKED PIPELINE
                # This provides persistent vehicle tracking across cameras and sessions
                processed_frame = pipeline.process(frame, topic)
                
                # Encode result back to bytes for Kafka
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result_bytes = buffer.tobytes()
                
                print(f"[UDF-ChromaDB] Processing completed for {topic} using ChromaDB-backed pipeline")
                
                # Clean up memory
                del frame_buffer, frame, processed_frame, buffer
                
                return result_bytes
                
            except Exception as e:
                print(f"[UDF-ChromaDB] ERROR processing frame from {topic}: {e}")
                import traceback
                traceback.print_exc()
                import gc
                gc.collect()
                return value  # Return original bytes on error

        # Process frames using UDF with ChromaDB-backed pipeline
        processed_df = df \
            .selectExpr("CAST(key AS STRING) as key",
                       "CAST(topic as STRING) as topic",
                       "value") \
            .withColumn("value", process_frame_with_chromadb("value", "topic"))

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
            .option("checkpointLocation", "tmp/cam1_processed_chromadb") \
            .outputMode("append") \
            .start()

        query_topic2 = processed_df \
            .filter("topic = 'cam2'") \
            .select("key", "value") \
            .writeStream \
            .format("kafka") \
            .options(**write_params[1]) \
            .option("checkpointLocation", "tmp/cam2_processed_chromadb") \
            .outputMode("append") \
            .start()

        try:
            # Wait for both queries to terminate
            print("[SPARK-ChromaDB] Starting vehicle ReID streaming with ChromaDB backend...")
            query_topic1.awaitTermination()
            query_topic2.awaitTermination()
        except KeyboardInterrupt:
            print("[SPARK-ChromaDB] KeyboardInterrupt received, stopping queries...")
            query_topic1.stop()
            query_topic2.stop()
        except Exception as e:
            print(f"[SPARK-ChromaDB] Error during execution: {e}")
            query_topic1.stop()
            query_topic2.stop()
        finally:
            # Close pipeline and ChromaDB connection
            if _shared_pipeline and hasattr(_shared_pipeline, 'classifier'):
                if hasattr(_shared_pipeline.classifier, 'close'):
                    _shared_pipeline.classifier.close()
                    print("[SPARK-ChromaDB] ChromaDB connection closed")
            spark.stop()
            print("[SPARK-ChromaDB] Spark session stopped")

    thread = threading.Thread(target=spark_streaming_thread)
    thread.start()
    return thread
