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
import gc
from realtime_reid.pipeline import Pipeline
from realtime_reid.vehicle_detector import VehicleDetector
from realtime_reid.feature_extraction import VehicleDescriptor
from realtime_reid.classifier_chromadb import ChromaDBVehicleReID

# Configure Python's garbage collector for better memory management
gc.set_threshold(100, 5, 5)

# Global pipeline instance with thread safety
_shared_pipeline = None
_current_session_id = None
_pipeline_lock = threading.Lock()

def get_or_create_shared_pipeline(session_id=None, base_dir=None):
    """Get or create the shared pipeline instance with ChromaDB backend."""
    global _shared_pipeline, _current_session_id, _pipeline_lock
    
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
    else:
        base_dir = Path(base_dir)

    with _pipeline_lock:
        if _shared_pipeline is None or _current_session_id != session_id:
            print(f"[Pipeline] Creating pipeline for session {session_id}")
            _current_session_id = session_id
            
            # Clean up old pipeline if exists
            if _shared_pipeline is not None:
                try:
                    if hasattr(_shared_pipeline, 'classifier') and hasattr(_shared_pipeline.classifier, 'close'):
                        _shared_pipeline.classifier.close()
                    _shared_pipeline = None
                    gc.collect()
                except Exception as e:
                    print(f"[Pipeline] Warning: Error during cleanup: {e}")
            
            try:
                # Model paths
                detector_path = base_dir / 'best_20.pt'
                descriptor_path = base_dir / 'best_osnet_model.pth'
                db_path = base_dir / f"chroma_vehicle_reid_streaming/{session_id}" if session_id else base_dir / "chroma_vehicle_reid_streaming"

                # Create components
                detector = VehicleDetector(model_path=str(detector_path))
                descriptor = VehicleDescriptor(model_type='osnet', model_path=str(descriptor_path))
                classifier = ChromaDBVehicleReID(
                    db_path=str(db_path),
                    collection_name=f"vehicle_embeddings_streaming{f'_{session_id}' if session_id else ''}"
                )
                
                # Create pipeline
                _shared_pipeline = Pipeline(detector=detector, descriptor=descriptor, classifier=classifier)
                print("[Pipeline] Created successfully")
                
                # Show database statistics
                stats = classifier.get_statistics()
                print(f"[Pipeline] Database: {stats['total']} embeddings, Max IDs: {stats['max_ids']}")
                
                gc.collect()
                
            except Exception as e:
                print(f"[Pipeline] ERROR creating pipeline: {e}")
                traceback.print_exc()
                _shared_pipeline = None
                raise
                
    return _shared_pipeline

def start_spark(session_id=None, base_dir=None):
    """Start Spark Streaming with ChromaDB vehicle re-identification."""
    
    # Spark/Kafka configuration
    SCALA_VERSION = '2.13'
    SPARK_VERSION = '4.0.0'
    KAFKA_VERSION = '3.5.0'
    
    print(f"Starting Spark Streaming for session: {session_id or 'default'}")
    
    packages = [
        f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
        f'org.apache.kafka:kafka-clients:{KAFKA_VERSION}'
    ]

    findspark.init()
    
    def spark_streaming_thread():
        print("[SPARK] Starting streaming thread")
        
        try:
            # Create Spark session
            spark = SparkSession.builder \
                .master('local[4]') \
                .appName(f"vehicle-reid-{session_id}" if session_id else "vehicle-reid") \
                .config("spark.jars.packages", ",".join(packages)) \
                .config("spark.sql.adaptive.enabled", "false") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.python.worker.memory", "1g") \
                .config("spark.driver.memory", "2g") \
                .config("spark.executor.memory", "2g") \
                .config("spark.python.worker.reuse", "true") \
                .getOrCreate()
            
            print("[SPARK] Session created successfully")
            
            # Test pipeline creation
            try:
                test_pipeline = get_or_create_shared_pipeline(session_id, base_dir)
                print("[SPARK] Pipeline test successful")
            except Exception as e:
                print(f"[SPARK] Pipeline creation failed: {e}")
                raise
            
            # Define Kafka parameters with better settings
            kafka_params = {
                "kafka.bootstrap.servers": "localhost:9092",
                "subscribe": f"cam1_{session_id},cam2_{session_id}" if session_id else "cam1,cam2",
                "maxOffsetsPerTrigger": "10",  # Reduced batch size to prevent memory spikes
                "fetchOffset.numRetries": "5",  # Increased retries for better reliability
                "kafka.consumer.fetch.max.bytes": "52428800",  # 50MB max fetch size
                "kafka.fetch.message.max.bytes": "52428800",   # 50MB max message size
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
                frame_counter = 0
                last_gc_time = time.time()
                
                @udf(BinaryType())
                def process_frame_with_chromadb(value, topic):
                    """Process individual frames using ChromaDB-backed vehicle ReID pipeline."""
                    nonlocal frame_counter, last_gc_time
                    frame_counter += 1
                    current_time = time.time()
                    
                    try:    
                        # Create pipeline instance inside UDF
                        pipeline = get_or_create_shared_pipeline(session_id, base_dir_str)
                        
                        # Decode bytes to OpenCV frame
                        frame_buffer = np.frombuffer(value, dtype=np.uint8)
                        frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            print(f"[UDF-ChromaDB] ERROR: Could not decode frame from {topic}")
                            return value
                        
                        # Log progress occasionally
                        if frame_counter % 100 == 0:
                            print(f"[UDF-ChromaDB] Processed {frame_counter} frames from {topic}")
                        
                        # Extract camera ID from topic
                        camera_id = 'cam1' if 'cam1' in topic else 'cam2'
                        
                        # Process frame through pipeline
                        result = pipeline.process(frame, camera_id=camera_id)
                        
                        # Periodic garbage collection
                        if current_time - last_gc_time > 60:  # Every minute
                            import gc
                            gc.collect()
                            last_gc_time = current_time
                        
                        return value
                        
                    except Exception as e:
                        print(f"[UDF-ChromaDB] ERROR processing frame: {e}")
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
                .option("truncate", "false") \
                .option("failOnDataLoss", "false") \
                .outputMode("append") \
                .trigger(processingTime="3 seconds") \
                .start()

            query_topic2 = processed_df \
                .filter(f"topic = 'cam2_{session_id}'" if session_id else "topic = 'cam2'") \
                .select("key", "value") \
                .writeStream \
                .format("kafka") \
                .options(**write_params[1]) \
                .option("truncate", "false") \
                .option("failOnDataLoss", "false") \
                .outputMode("append") \
                .trigger(processingTime="3 seconds") \
                .start()

            try:
                # Wait for both queries to process all data with a timeout mechanism
                print("[SPARK-ChromaDB] Starting vehicle ReID streaming with ChromaDB backend...")
                
                # Log initial memory usage
                print("[SPARK-ChromaDB] Initial processing started")
                
                # Keep track of last activity time and memory check time
                last_active_time = time.time()
                last_gc_time = time.time()
                idle_timeout = 300  # 5 minutes idle timeout
                gc_interval = 120  # Force GC every 2 minutes
                
                # Simplified monitoring loop
                while True:
                    time.sleep(10)  # Check every 10 seconds
                    
                    # Check if both queries are still active
                    if not query_topic1.isActive and not query_topic2.isActive:
                        print("[SPARK-ChromaDB] Both queries completed. Stopping.")
                        break
                    
                    # Update activity time if either query is active
                    if query_topic1.isActive or query_topic2.isActive:
                        last_active_time = time.time()
                    
                    # Check for idle timeout
                    if time.time() - last_active_time > idle_timeout:
                        print(f"[SPARK-ChromaDB] Idle timeout reached ({idle_timeout}s). Stopping.")
                        break
                    
                    # Periodic garbage collection
                    if time.time() - last_gc_time > gc_interval:
                        print("[SPARK-ChromaDB] Running garbage collection")
                        gc.collect()
                        last_gc_time = time.time()
                
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
                try:
                    # Release resources from pipeline
                    if _shared_pipeline:
                        # First close ChromaDB connection
                        if hasattr(_shared_pipeline, 'classifier'):
                            if hasattr(_shared_pipeline.classifier, 'close'):
                                try:
                                    _shared_pipeline.classifier.close()
                                    print("[SPARK-ChromaDB] ChromaDB connection closed")
                                except Exception as e:
                                    print(f"[SPARK-ChromaDB] Warning: Error closing ChromaDB: {e}")
                        
                        # Release other resources
                        if hasattr(_shared_pipeline, 'detector'):
                            _shared_pipeline.detector = None
                        if hasattr(_shared_pipeline, 'descriptor'):
                            _shared_pipeline.descriptor = None
                        
                        # Reset pipeline reference
                        _shared_pipeline = None
                        
                    # Force explicit garbage collection
                    print("[SPARK-ChromaDB] Running final garbage collection")
                    gc.collect()
                    
                    # Stop Spark session
                    spark.stop()
                    print("[SPARK-ChromaDB] Spark session stopped")
                    
                except Exception as e:
                    print(f"[SPARK-ChromaDB] Error during cleanup: {e}")
                    traceback.print_exc()
                
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