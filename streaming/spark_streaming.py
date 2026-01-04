import threading
import numpy as np
import cv2
import findspark
import shutil
import json
import base64
import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType
from pyspark.sql.streaming import StreamingQueryListener
from pathlib import Path
import time
import os
import traceback
import gc
from modules import color
from kafka import KafkaProducer
from modules.pipeline import Pipeline_spark
from modules.vehicle_detection import VehicleDetector
from modules.feature_extraction import VehicleDescriptor
from modules.reid_chromadb import ChromaDBVehicleReID_spark

# Configure Python's garbage collector for better memory management
gc.set_threshold(100, 5, 5)

_model_state_bc = None
_shared_model = None
_kafka_producer = None
_processed_cross_camera = set()
_producer_lock = threading.Lock()


def _parse_capture_ts_ms_from_key(key_value):
    if key_value is None:
        return None
    try:
        key_str = key_value
        if isinstance(key_value, (bytes, bytearray, memoryview)):
            key_str = bytes(key_value).decode("utf-8", errors="ignore")
        else:
            key_str = str(key_value)
        ts_part = key_str.split(":", 1)[0].strip()
        if not ts_part:
            return None
        return int(ts_part)
    except Exception:
        return None


def get_or_create_kafka_producer():
    global _kafka_producer
    with _producer_lock:
        if _kafka_producer is None:
            _kafka_producer = KafkaProducer(
                bootstrap_servers="localhost:9092",
                linger_ms=5,
                max_in_flight_requests_per_connection=1,
            )
            print("[KafkaProducer] Created shared producer")
    return _kafka_producer


def close_kafka_producer():
    global _kafka_producer
    with _producer_lock:
        if _kafka_producer is None:
            return
        try:
            _kafka_producer.flush()
        except Exception as flush_error:
            print(f"[KafkaProducer] Flush warning: {flush_error}")
        finally:
            try:
                _kafka_producer.close()
                print("[KafkaProducer] Closed shared producer")
            except Exception as close_error:
                print(f"[KafkaProducer] Close warning: {close_error}")
            _kafka_producer = None


def get_shared_model(model_state_bc):
    global _shared_model
    try:   
        state = model_state_bc.value
        if state is None:
            raise ValueError("Model state is not available in broadcast variable")
        if _shared_model is None:
            detector = VehicleDetector(model_path=state["detector_path"])
            descriptor = VehicleDescriptor(model_type='osnet', model_path=state["descriptor_path"])
            _shared_model = Pipeline_spark(detector=detector, descriptor=descriptor)
            print("[Pipeline] Created successfully")
        else:
            print("[Pipeline] Reusing existing instance")
    except Exception as e:
        print(f"[Pipeline] ERROR creating pipeline: {e}")
        traceback.print_exc()
        gc.collect()
        _shared_model = None
        raise e
    return _shared_model

def _resolve_camera_from_topic(topic_value) -> str:
    if topic_value is None:
        return "unknown"
    if isinstance(topic_value, (bytes, bytearray, memoryview)):
        topic_str = bytes(topic_value).decode("utf-8", errors="ignore")
    else:
        topic_str = str(topic_value)
    topic_root = topic_str.split("_")[0].strip()
    return topic_root if topic_root else "unknown"


def _serialize_payloads(payloads) -> str:
    records = []
    for payload in payloads:
        try:
            record = {
                "camera_id": payload.camera_id,
                "track_id": payload.track_id,
                "vehicle_type": payload.vehicle_type,
                "confidence": payload.confidence,
                "timestamp": payload.timestamp,
                "bbox": payload.bbox,
                "image": base64.b64encode(payload.thumbnail).decode("utf-8") if payload.thumbnail else None,
                "feature": payload.feature.tolist() if hasattr(payload.feature, "tolist") else list(payload.feature),
                "thumbnail": base64.b64encode(payload.thumbnail).decode("utf-8") if payload.thumbnail else None,
            }
            records.append(record)
        except Exception as serialization_error:
            print(f"[UDF] Payload serialization error: {serialization_error}")
    return json.dumps(records)


def _upsert_payloads(raw_frame, payloads_json: str, classifier: ChromaDBVehicleReID_spark):
    VEHICLE_LABELS = {0: 'motorcycle', 1: 'car', 2: 'truck', 3: 'bus'}
    if not payloads_json or payloads_json == "[]":
        return raw_frame
    try:
        payloads = json.loads(payloads_json)
    except json.JSONDecodeError as decode_error:
        print(f"[Payload] JSON decode error: {decode_error}")
        return raw_frame
    annotated_frame = raw_frame
    for payload in payloads:
        try:
            encoded = payload.get("image")
            decoded_image = None
            if encoded:
                try:
                    img_bytes = base64.b64decode(encoded)
                    img_buffer = np.frombuffer(img_bytes, dtype=np.uint8)
                    decoded_image = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
                except Exception as decode_err:
                    print(f"[Payload] Image decode error: {decode_err}")
            feature = torch.tensor(payload.get("feature", []), dtype=torch.float32)
            vehicle_id = classifier.identify(
                target=feature,
                vehicle_type=int(payload.get("vehicle_type")),
                confidence=float(payload.get("confidence")),
                do_update=True,
                image=decoded_image,
                camera_id=payload.get("camera_id"),
                track_id=payload.get("track_id"),
                timestamp=payload.get("timestamp"),
            )

            camera_matches = classifier.get_cross_camera_matches(vehicle_id, payload.get("vehicle_type"))
            if len(camera_matches) > 1:
                vehicle_key = (payload.get("camera_id"), vehicle_id)
                vehicle_type_id = int(payload.get("vehicle_type"))
                vehicle_label = VEHICLE_LABELS.get(vehicle_type_id, f"class_{vehicle_type_id}")
                print(f"CROSS-CAMERA MATCH! Vehicle ID: {vehicle_id} ({vehicle_label}) found in cameras: {list(camera_matches.keys())}")
            
                if vehicle_key not in _processed_cross_camera:
                    classifier.save_cross_camera_images(vehicle_id, payload.get("vehicle_type"), camera_matches, save_dir="Vehicle-Re-ID/matching_exports_2")
                    _processed_cross_camera.add(vehicle_key)
            vehicle_type = VEHICLE_LABELS.get(int(payload.get("vehicle_type")), f"class_{payload.get('vehicle_type')}")
            label = f"ID: {vehicle_id} ({vehicle_type})"
            unique_color = color.create_unique_color(vehicle_id)
            cv2.rectangle(
                    img=annotated_frame,
                    pt1=(payload.get("bbox")[0], payload.get("bbox")[1]),
                    pt2=(payload.get("bbox")[2], payload.get("bbox")[3]),
                    color=unique_color,
                    thickness=2,
                )
            cv2.putText(
                img=annotated_frame,
                text=label,
                org=(payload.get("bbox")[0], payload.get("bbox")[1] - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=unique_color,
                thickness=2,
            )
        except Exception as upsert_error:
            print(f"[Payload] Upsert error: {upsert_error}")
    return annotated_frame

def start_spark_streaming(args=None, base_dir=None, progress_callback=None, session_metrics=None):
    global _model_state_bc, _shared_model, _kafka_producer, _processed_cross_camera
    try:
        findspark.init()
        SCALA_VERSION = '2.13'
        SPARK_VERSION = '4.0.0'
        KAFKA_VERSION = '3.5.0'
        
        print(f"Starting Spark Streaming for Vehicle Re-ID...")
        
        packages = [
            f'org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}',
            f'org.apache.kafka:kafka-clients:{KAFKA_VERSION}'
        ]

        spark = SparkSession.builder \
            .master('local[4]') \
            .appName(f"vehicle-reid") \
            .config("spark.jars.packages", ",".join(packages)) \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
            .config("spark.python.worker.faulthandler.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.python.worker.memory", "3g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.dynamicAllocation.enabled", "false") \
            .config("spark.executor.memory", "3g") \
            .config("spark.python.worker.reuse", "true") \
            .getOrCreate()
        
        print("[SPARK] Session created successfully")
        base_dir = Path(base_dir or Path(__file__).resolve().parents[1])

        detector = VehicleDetector(model_path=str(base_dir / "best_20.pt"))
        descriptor = VehicleDescriptor(model_type='osnet', model_path=str(base_dir / "best_osnet_model.pth"))
        pipeline = Pipeline_spark(detector=detector, descriptor=descriptor)
        print(f"[Driver] Pipeline initialized once. Pid={os.getpid()}")

        listener = None
        if progress_callback:
            class _MetricsListener(StreamingQueryListener):
                def onQueryStarted(self_inner, event):
                    pass

                def onQueryTerminated(self_inner, event):
                    pass

                def onQueryProgress(self_inner, event):
                    try:
                        progress = event.progress
                        duration_ms = None
                        if progress.durationMs:
                            duration_ms = progress.durationMs.get('addBatch') or progress.durationMs.get('getBatch')
                        payload = {
                            "batch_id": progress.batchId,
                            "num_input_rows": progress.numInputRows,
                            "processed_rows_per_second": progress.processedRowsPerSecond,
                            "duration_ms": duration_ms,
                        }
                        progress_callback(payload)
                    except Exception as callback_error:
                        print(f"[SPARK] Metrics callback error: {callback_error}")

            listener = _MetricsListener()
            spark.streams.addListener(listener)
        
        kafka_params = {
            "kafka.bootstrap.servers": "localhost:9092",
            "subscribe": f"cam1, cam2",
            "maxOffsetsPerTrigger": "100",  
            "fetchOffset.numRetries": "2",  
            "kafka.consumer.fetch.max.bytes": "52428800",  
            "kafka.fetch.message.max.bytes": "52428800",  
            "fetchOffset.retryIntervalMs": "1000",
            "kafka.session.timeout.ms": "30000",
            "kafka.request.timeout.ms": "40000",
            "kafka.max.poll.interval.ms": "300000"
        }
        
        print(f"[SPARK] Kafka parameters: {kafka_params}")

        print(f"[SPARK] Creating streaming DataFrame...")
        df = spark.readStream \
            .format("kafka") \
            .options(**kafka_params) \
            .option("startingOffsets", "latest") \
            .load()
        
        print(f"[SPARK] Streaming DataFrame created")
        
        df = df.withColumn("value", df["value"].cast(BinaryType()))
    
        checkpoint_base = Path(base_dir) / "tmp" if base_dir else Path("tmp")
        checkpoint_base.mkdir(exist_ok=True)

        checkpoint_dir = checkpoint_base / (f"chromadb_checkpoint")
        if checkpoint_dir.exists():
            try:
                shutil.rmtree(checkpoint_dir)
                print(f"[SPARK-ChromaDB] Cleaned up old checkpoint directory: {checkpoint_dir}")
            except Exception as e:
                print(f"[SPARK-ChromaDB] Warning: Could not clean checkpoint dir {checkpoint_dir}: {e}")

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        

        db_root = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
        classifier = ChromaDBVehicleReID_spark(
            db_path=str(db_root / "chroma_vehicle_reid_streaming"),
            collection_name="vehicle_embeddings_streaming"
        )
        print("[ChromaDB] Connected for batch upserts")

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        stats_start_ts = time.time()
        per_camera_totals = {
            "cam1": {"processed": 0, "latency_ms_sum": 0.0},
            "cam2": {"processed": 0, "latency_ms_sum": 0.0},
        }

        per_camera_times = {
            "cam1": {
                "first_capture_ts_ms": None,
                "last_capture_ts_ms": None,
                "first_spark_output_ts_ms": None,
                "last_spark_output_ts_ms": None,
            },
            "cam2": {
                "first_capture_ts_ms": None,
                "last_capture_ts_ms": None,
                "first_spark_output_ts_ms": None,
                "last_spark_output_ts_ms": None,
            },
        }
        def process_result_batch(batch_df, batch_id):
            print(f"[ForeachBatch] Handling batch {batch_id}")
            producer = get_or_create_kafka_producer()
            try:
                
                batch_per_camera = {
                    "cam1": {"processed": 0, "latency_ms_sum": 0.0},
                    "cam2": {"processed": 0, "latency_ms_sum": 0.0},
                }
                batch_duration_s = 0.0
                for row in batch_df.selectExpr(
                    "CAST(topic AS STRING) as topic",
                    "CAST(key AS STRING) as key",
                    "value",
                    "timestamp",
                ).toLocalIterator():
                    camera_id = _resolve_camera_from_topic(row.topic)

                    capture_ts_ms = _parse_capture_ts_ms_from_key(getattr(row, "key", None))
                    if camera_id in per_camera_times and capture_ts_ms is not None:
                        t = per_camera_times[camera_id]
                        if t["first_capture_ts_ms"] is None or capture_ts_ms < t["first_capture_ts_ms"]:
                            t["first_capture_ts_ms"] = capture_ts_ms
                        if t["last_capture_ts_ms"] is None or capture_ts_ms > t["last_capture_ts_ms"]:
                            t["last_capture_ts_ms"] = capture_ts_ms

                    frame_bytes = bytes(row.value) if isinstance(row.value, (bytearray, memoryview)) else row.value
                    if frame_bytes is None:
                        continue

                    image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        continue

                    batch_start_ts = time.time()
                    pipeline_result = pipeline.process(image, camera_id=camera_id)
                    batch_duration_s = batch_duration_s + max(1e-6, time.time() - batch_start_ts)
                    if isinstance(pipeline_result, tuple) and len(pipeline_result) == 2:
                        raw_frame, payloads = pipeline_result
                    else:
                        raw_frame, payloads = image, pipeline_result

                    if raw_frame is None or getattr(raw_frame, "size", 0) == 0:
                        raw_frame = image

                    try:
                        now_s = time.time()
                        kafka_ts = row.timestamp
                        if capture_ts_ms is not None:
                            e2e_latency_ms = (now_s * 1000.0) - float(capture_ts_ms)
                        elif kafka_ts is not None:
                            e2e_latency_ms = (now_s - kafka_ts.timestamp()) * 1000.0
                        else:
                            e2e_latency_ms = None

                        if e2e_latency_ms is not None:
                            if camera_id in batch_per_camera:
                                batch_per_camera[camera_id]["processed"] += 1
                                batch_per_camera[camera_id]["latency_ms_sum"] += e2e_latency_ms
                                per_camera_totals[camera_id]["processed"] += 1
                                per_camera_totals[camera_id]["latency_ms_sum"] += e2e_latency_ms
                    except Exception:
                        pass
                         
                    payloads_json = _serialize_payloads(payloads)
                    annotated_image = _upsert_payloads(raw_frame, payloads_json, classifier)

                    # Guard against OpenCV assertion failures on empty images.
                    if annotated_image is None or getattr(annotated_image, "size", 0) == 0:
                        annotated_image = raw_frame
                    if annotated_image is None or getattr(annotated_image, "size", 0) == 0:
                        continue

                    success, buffer = cv2.imencode('.jpg', annotated_image, encode_params)
                    if success:
                        spark_out_ts_ms = time.time() * 1000.0
                        if camera_id in per_camera_times:
                            t = per_camera_times[camera_id]
                            if t["first_spark_output_ts_ms"] is None or spark_out_ts_ms < t["first_spark_output_ts_ms"]:
                                t["first_spark_output_ts_ms"] = spark_out_ts_ms
                            if t["last_spark_output_ts_ms"] is None or spark_out_ts_ms > t["last_spark_output_ts_ms"]:
                                t["last_spark_output_ts_ms"] = spark_out_ts_ms
                        producer.send(f"{camera_id}_processed", value=buffer.tobytes())
                producer.flush()

                if progress_callback:
                    try:
                        progress_callback(
                            {
                                "type": "per_camera_metrics",
                                "batch_id": batch_id,
                                "batch_duration_s": batch_duration_s,
                                "per_camera": {
                                    cam: {
                                        "batch_processed": batch_per_camera[cam]["processed"],
                                        "batch_mean_e2e_latency_ms": (
                                            (batch_per_camera[cam]["latency_ms_sum"] / batch_per_camera[cam]["processed"])
                                            if batch_per_camera[cam]["processed"]
                                            else None
                                        ),
                                        "total_processed": per_camera_totals[cam]["processed"],
                                        "total_mean_e2e_latency_ms": (
                                            (per_camera_totals[cam]["latency_ms_sum"] / per_camera_totals[cam]["processed"])
                                            if per_camera_totals[cam]["processed"]
                                            else None
                                        ),
                                        "first_capture_ts_ms": per_camera_times[cam]["first_capture_ts_ms"],
                                        "last_capture_ts_ms": per_camera_times[cam]["last_capture_ts_ms"],
                                        "first_spark_output_ts_ms": per_camera_times[cam]["first_spark_output_ts_ms"],
                                        "last_spark_output_ts_ms": per_camera_times[cam]["last_spark_output_ts_ms"],
                                    }
                                    for cam in ("cam1", "cam2")
                                },
                            }
                        )
                    except Exception as callback_error:
                        print(f"[SPARK] Metrics callback error: {callback_error}")

                print(f"[ForeachBatch] Finished batch {batch_id} of {batch_df.count()} frames, time: {batch_duration_s:.2f}s")
            except Exception as batch_error:
                print(f"[ForeachBatch] Error: {batch_error}")
                traceback.print_exc()
                producer.flush()

        query = df.writeStream \
            .foreachBatch(process_result_batch) \
            .option("checkpointLocation", str(checkpoint_dir)) \
            .trigger(processingTime="8 seconds") \
            .start()
        gc.collect()
        query.awaitTermination()

    except Exception as e:
        print(f"[SPARK] ERROR creating Spark session: {e}")
        traceback.print_exc()
        gc.collect()
        return None
    finally:
        close_kafka_producer()