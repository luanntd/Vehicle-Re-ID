import threading
import numpy as np
import cv2
import findspark
import os
import shutil
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType
from pathlib import Path
import time
import traceback
import shutil
import os
import sys
import gc
import numpy as np
import findspark
import cv2
from realtime_reid.pipeline import Pipeline
from realtime_reid.vehicle_detector import VehicleDetector
from realtime_reid.feature_extraction import VehicleDescriptor
from realtime_reid.classifier_chromadb import ChromaDBVehicleReID
import gc

# Global variable to hold the single shared pipeline instance with ChromaDB
# This will be initialized lazily to avoid serialization
_shared_pipeline = None
_current_session_id = None

def get_or_create_shared_pipeline(session_id=None, base_dir=None):
    """Get or create the shared pipeline instance with ChromaDB backend."""
    global _shared_pipeline, _current_session_id
    
    # If base_dir is not provided, determine it from the current file's location.
    # This makes the function usable in non-Spark contexts as well.
    if base_dir is None:
        from pathlib import Path
        base_dir = Path(__file__).resolve().parents[2] # Assuming Vehicle-Re-ID is the root
    else:
        from pathlib import Path
        base_dir = Path(base_dir)

    # Create new pipeline if none exists or session_id changed
    if _shared_pipeline is None or _current_session_id != session_id:
        print(f"[Pipeline Init] Creating CHROMADB-INTEGRATED PIPELINE for session {session_id}")
        _current_session_id = session_id
        
        try:
            # Use absolute paths for models to ensure they are found
            detector_path = base_dir / 'best_20.pt'
            descriptor_path = base_dir / 'best_osnet_model.pth'
            db_path = base_dir / f"chroma_vehicle_reid_streaming/{session_id}" if session_id else base_dir / "chroma_vehicle_reid_streaming"

            print(f"[Pipeline Init] Detector Path: {detector_path}")
            print(f"[Pipeline Init] Descriptor Path: {descriptor_path}")
            print(f"[Pipeline Init] DB Path: {db_path}")

            detector = VehicleDetector(model_path=str(detector_path))
            descriptor = VehicleDescriptor(model_type='osnet', model_path=str(descriptor_path))
            
            # Initialize ChromaDB classifier with a persistent, absolute path
            chromadb_classifier = ChromaDBVehicleReID(
                db_path=str(db_path),
                collection_name=f"vehicle_embeddings_streaming{f'_{session_id}' if session_id else ''}"
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
            
        except Exception as e:
            print(f"[Pipeline Init] ERROR creating pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise
    return _shared_pipeline

def start_spark(session_id = None, base_dir = None):
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
    SCALA_VERSION = '2.13'  # Standard version for many Spark installations
    SPARK_VERSION = '4.0.0' # A common stable version
    KAFKA_VERSION = '3.5.0' # Match Kafka client to Spark for compatibility
    print("Starting Spark Streaming on", session_id if session_id else "default session")
    packages = [
        f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
        f'org.apache.kafka:kafka-clients:{KAFKA_VERSION}'
    ]

    findspark.init()
    _shared_pipeline = None
    _current_session_id = None
    
    # Frame counter for garbage collection
    frame_counter = 0
    
    def spark_streaming_thread():
        print(f"[SPARK] Starting Spark streaming thread for session: {session_id}")
        
        try:
            # Initialize Spark session first
            print(f"[SPARK] Initializing Spark session...")
            spark = SparkSession.builder \
                .master('local[2]') \
                .appName(f"vehicle-reid-chromadb-{session_id}" if session_id else "vehicle-reid-chromadb") \
                .config("spark.jars.packages", ",".join(packages)) \
                .config("spark.sql.adaptive.enabled", "false") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .config("spark.serializer.objectStreamReset", "100") \
                .config("spark.rdd.compress", "false") \
                .config("spark.kryo.unsafe", "true") \
                .config("spark.kryoserializer.buffer.max", "512m") \
                .config("spark.sql.streaming.checkpointLocation", f"/tmp/spark-checkpoint-{session_id}") \
                .config("spark.cleaner.periodicGC.interval", "10s") \
                .config("spark.cleaner.referenceTracking.blocking", "false") \
                .getOrCreate()
            
            print(f"[SPARK] Spark session created successfully")
            
            spark.conf.set("spark.sql.shuffle.partitions", "2")  # Slightly increase partitions
            
            # Test pipeline creation early
            print(f"[SPARK] Testing pipeline creation...")
            try:
                test_pipeline = get_or_create_shared_pipeline(session_id, base_dir)
                print(f"[SPARK] Pipeline test successful, ChromaDB initialized")
            except Exception as e:
                print(f"[SPARK] ERROR: Pipeline creation failed: {e}")
                raise
            
            # Define Kafka parameters with better settings
            kafka_params = {
                "kafka.bootstrap.servers": "localhost:9092",
                "subscribe": f"cam1_{session_id},cam2_{session_id}" if session_id else "cam1,cam2",
                "maxOffsetsPerTrigger": "20",  # Increased batch size
                "fetchOffset.numRetries": "3",
                "fetchOffset.retryIntervalMs": "1000",
                "kafka.session.timeout.ms": "30000",
                "kafka.request.timeout.ms": "40000",
                "kafka.max.poll.interval.ms": "300000"
            }
            
            print(f"[SPARK] Kafka parameters: {kafka_params}")
            
            # Create streaming DataFrame with Kafka source
            print(f"[SPARK] Creating streaming DataFrame...")
            df = spark.readStream \
                .format("kafka") \
                .options(**kafka_params) \
                .option("startingOffsets", "latest") \
                .load()
            
            print(f"[SPARK] Streaming DataFrame created")
            
            df = df.withColumn("value", df["value"].cast(BinaryType()))
            
            # UDF for processing frames with ChromaDB-backed pipeline
            def create_process_frame_udf(session_id, base_dir_str):
                frame_counter = 0  # Local counter for each UDF instance
                
                @udf(BinaryType())
                def process_frame_with_chromadb(value, topic):
                    """Process individual frames using ChromaDB-backed vehicle ReID pipeline."""
                    nonlocal frame_counter
                    frame_counter += 1
                    
                    try:    
                        # Create pipeline instance inside UDF to avoid serialization issues
                        pipeline = get_or_create_shared_pipeline(session_id, base_dir_str)
                        
                        # Decode bytes to OpenCV frame
                        frame_buffer = np.frombuffer(value, dtype=np.uint8)
                        frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            print(f"[UDF-ChromaDB] ERROR: Could not decode frame from {topic}")
                            return value
                        
                        # Only log every 50th frame to reduce console spam
                        if frame_counter % 50 == 0:
                            print(f"[UDF-ChromaDB] Processing frame {frame_counter} from {topic}")
                        
                        # Process frame with CHROMADB-BACKED PIPELINE
                        processed_frame = pipeline.process(frame, topic[4:])
                        
                        # Encode result back to bytes for Kafka
                        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        result_bytes = buffer.tobytes()
                        
                        # Aggressive memory cleanup
                        del frame_buffer, frame, processed_frame, buffer
                        
                        # More frequent garbage collection for UDF
                        if frame_counter % 25 == 0:
                            gc.collect()
                        
                        return result_bytes
                        
                    except Exception as e:
                        print(f"[UDF-ChromaDB] ERROR processing frame from {topic}: {e}")
                        # Force garbage collection on error
                        gc.collect()
                        return value
                
                return process_frame_with_chromadb
            
            # Create the UDF with session_id
            print(f"[SPARK] Creating UDF...")
            process_frame_udf = create_process_frame_udf(session_id, str(base_dir) if base_dir else None)
            
            # Process frames using UDF with ChromaDB-backed pipeline
            print(f"[SPARK] Creating processed DataFrame...")
            processed_df = df \
                .selectExpr("CAST(key AS STRING) as key",
                           "CAST(topic as STRING) as topic",
                           "value") \
                .withColumn("value", process_frame_udf("value", "topic"))

            # Define a base for checkpoint locations, using the base_dir if available
            checkpoint_base = Path(base_dir) / "tmp" if base_dir else Path("tmp")
            checkpoint_base.mkdir(exist_ok=True)
            
            # Clean up old checkpoint directories
            checkpoint_dirs = [
                checkpoint_base / f"cam1_processed_chromadb_{session_id}" if session_id else checkpoint_base / "cam1_processed_chromadb",
                checkpoint_base / f"cam2_processed_chromadb_{session_id}" if session_id else checkpoint_base / "cam2_processed_chromadb"
            ]
            
            for checkpoint_dir in checkpoint_dirs:
                if checkpoint_dir.exists():
                    try:
                        shutil.rmtree(checkpoint_dir)
                        print(f"[SPARK-ChromaDB] Cleaned up old checkpoint directory: {checkpoint_dir}")
                    except Exception as e:
                        print(f"[SPARK-ChromaDB] Warning: Could not clean checkpoint dir {checkpoint_dir}: {e}")

            write_params = [
                {
                    "kafka.bootstrap.servers": "localhost:9092",
                    "topic": f"cam1_processed_{session_id}" if session_id else "cam1_processed",
                    "checkpointLocation": str(checkpoint_dirs[0]),
                },
                {
                    "kafka.bootstrap.servers": "localhost:9092", 
                    "topic": f"cam2_processed_{session_id}" if session_id else "cam2_processed",
                    "checkpointLocation": str(checkpoint_dirs[1]),
                }
            ]
            
            # Write processed frames back to Kafka with better trigger settings
            print(f"[SPARK] Starting streaming queries...")
            query_topic1 = processed_df \
                .filter(f"topic = 'cam1_{session_id}'" if session_id else "topic = 'cam1'") \
                .select("key", "value") \
                .writeStream \
                .format("kafka") \
                .options(**write_params[0]) \
                .outputMode("append") \
                .trigger(processingTime="2 seconds") \
                .start()

            query_topic2 = processed_df \
                .filter(f"topic = 'cam2_{session_id}'" if session_id else "topic = 'cam2'") \
                .select("key", "value") \
                .writeStream \
                .format("kafka") \
                .options(**write_params[1]) \
                .outputMode("append") \
                .trigger(processingTime="2 seconds") \
                .start()

            try:
                # Wait for both queries to process all data with a timeout mechanism
                print("[SPARK-ChromaDB] Starting vehicle ReID streaming with ChromaDB backend...")
                
                # Keep track of last activity time
                last_active_time = time.time()
                idle_timeout = 300  # 5 minutes idle timeout before considering processing complete
                activity_check_interval = 30  # Check for activity every 30 seconds
                
                # Create a flag accessible to the main thread to signal termination
                global_shutdown = [False]
                
                # Set up a shutdown hook for clean termination from outside
                def shutdown_hook():
                    global_shutdown[0] = True
                    
                # Track progress and detect when processing is complete
                while not global_shutdown[0]:
                    # Check if queries are active
                    recent_progress1 = query_topic1.recentProgress
                    recent_progress2 = query_topic2.recentProgress
                    
                    has_activity = False
                    
                    # Log status every 30 seconds
                    print(f"[SPARK-ChromaDB] Status: Query 1 progress entries: {len(recent_progress1)}, Query 2 progress entries: {len(recent_progress2)}")
                    
                    # Check for input rows processed in recent batches
                    if recent_progress1 and any(p.get("numInputRows", 0) > 0 for p in recent_progress1):
                        print(f"[SPARK-ChromaDB] Query 1 active - processing data")
                        has_activity = True
                        last_active_time = time.time()
                        
                    if recent_progress2 and any(p.get("numInputRows", 0) > 0 for p in recent_progress2):
                        print(f"[SPARK-ChromaDB] Query 2 active - processing data")
                        has_activity = True
                        last_active_time = time.time()
                    
                    # If no activity for a while, check if we should shutdown
                    if not has_activity:
                        idle_time = time.time() - last_active_time
                        print(f"[SPARK-ChromaDB] No activity detected for {idle_time:.1f} seconds")
                        
                        if idle_time > idle_timeout:
                            print(f"[SPARK-ChromaDB] Idle timeout reached ({idle_timeout}s). Assuming processing is complete.")
                            break
                    
                    # Sleep before next check
                    time.sleep(activity_check_interval)
                
                # Stop queries gracefully
                print("[SPARK-ChromaDB] Stopping streaming queries...")
                query_topic1.stop()
                query_topic2.stop()
                print("[SPARK-ChromaDB] All streaming queries stopped")
                
            except KeyboardInterrupt:
                print("[SPARK-ChromaDB] KeyboardInterrupt received, stopping queries...")
                query_topic1.stop()
                query_topic2.stop()
            except Exception as e:
                print(f"[SPARK-ChromaDB] Error during execution: {e}")
                import traceback
                traceback.print_exc()
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
                
        except Exception as e:
            print(f"[SPARK] Fatal error in streaming thread: {e}")
            import traceback
            traceback.print_exc()
            if 'spark' in locals():
                spark.stop()
            print("[SPARK] Spark session stopped due to error")

    thread = threading.Thread(target=spark_streaming_thread)
    thread.start()
    return thread