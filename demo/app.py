from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
import os
import sys
import uuid
import threading
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import shutil
from pathlib import Path
import base64
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
import time
from queue import Queue
import numpy as np
import gc
import traceback

# Add the parent directory to Python path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global queue for WebSocket messages
websocket_queue = Queue()


class SessionMetrics:
    def __init__(self, session_id: str):
        self.lock = threading.Lock()
        self.session_id = session_id
        self.camera_metrics = {
            "cam1": {
                "fps_sum": 0.0,
                "fps_count": 0,
                "latency_sum": 0.0,
                "latency_count": 0,
                "frame_count": 0,
            },
            "cam2": {
                "fps_sum": 0.0,
                "fps_count": 0,
                "latency_sum": 0.0,
                "latency_count": 0,
                "frame_count": 0,
            },
        }

        # Spark-side capture->Spark-output metrics (separate from true e2e).
        self.spark_camera_metrics = {
            "cam1": {"frame_count": 0, "latency_sum": 0.0, "latency_count": 0},
            "cam2": {"frame_count": 0, "latency_sum": 0.0, "latency_count": 0},
        }

        self.spark_run_start: Optional[float] = None
        self.spark_run_end: Optional[float] = None
        self.spark_batch_count = 0

    def record_frame_latency(self, latency_ms: float, camera: Optional[str] = None):
        if latency_ms is None or camera not in self.camera_metrics:
            return
        with self.lock:
            stats = self.camera_metrics[camera]
            stats["latency_sum"] += latency_ms
            stats["latency_count"] += 1

    def record_frame_sent(self, camera: Optional[str] = None):
        if camera not in self.camera_metrics:
            return
        with self.lock:
            self.camera_metrics[camera]["frame_count"] += 1

    def record_spark_frame_metrics(self, camera: Optional[str], latency_ms: Optional[float] = None):
        if camera not in self.spark_camera_metrics:
            return
        with self.lock:
            stats = self.spark_camera_metrics[camera]
            stats["frame_count"] += 1
            if latency_ms is not None:
                stats["latency_sum"] += float(latency_ms)
                stats["latency_count"] += 1

    def record_frontend_fps(self, camera: str, fps: float):
        if not fps or camera not in self.camera_metrics:
            return
        with self.lock:
            stats = self.camera_metrics[camera]
            stats["fps_sum"] += fps
            stats["fps_count"] += 1

    def start_spark_run(self):
        with self.lock:
            if self.spark_run_start is None:
                self.spark_run_start = time.time()
                self.spark_run_end = None

    def stop_spark_run(self):
        with self.lock:
            if self.spark_run_start is not None:
                self.spark_run_end = time.time()

    def record_micro_batch(self, progress: dict):
        if not progress:
            return
        with self.lock:
            self.spark_batch_count += 1

    def compute_summary(self):
        with self.lock:
            cameras = {}
            spark_cameras = {}
            for camera, stats in self.camera_metrics.items():
                mean_fps = (stats["fps_sum"] / stats["fps_count"]) if stats["fps_count"] else None
                mean_latency_ms = (stats["latency_sum"] / stats["latency_count"]) if stats["latency_count"] else None
                cameras[camera] = {
                    # Mean FPS is computed from measured FPS samples.
                    "mean_fps": mean_fps,
                    # True end-to-end latency (capture -> websocket send).
                    "total_mean_e2e_latency_ms": mean_latency_ms,
                    "fps_samples": stats["fps_count"],
                    "latency_samples": stats["latency_count"],
                    "total_processed": stats.get("frame_count", 0),
                }

            if self.spark_run_start is None:
                spark_time = None
            else:
                end_time = self.spark_run_end or time.time()
                spark_time = max(0.0, end_time - self.spark_run_start)
            spark_batches = self.spark_batch_count

            # Add runtime-based mean e2e FPS (like main_stream.py).
            if spark_time and spark_time > 0:
                for camera in cameras.keys():
                    total = cameras[camera].get("total_processed") or 0
                    cameras[camera]["mean_e2e_fps"] = float(total) / float(spark_time)
            else:
                for camera in cameras.keys():
                    cameras[camera]["mean_e2e_fps"] = None

            # Spark-side capture->Spark-output metrics.
            for camera, stats in self.spark_camera_metrics.items():
                mean_spark_latency_ms = (stats["latency_sum"] / stats["latency_count"]) if stats["latency_count"] else None
                spark_cameras[camera] = {
                    "total_processed": stats.get("frame_count", 0),
                    "total_mean_e2e_latency_ms": mean_spark_latency_ms,
                }
            if spark_time and spark_time > 0:
                for camera in spark_cameras.keys():
                    total = spark_cameras[camera].get("total_processed") or 0
                    spark_cameras[camera]["mean_e2e_fps"] = float(total) / float(spark_time)
            else:
                for camera in spark_cameras.keys():
                    spark_cameras[camera]["mean_e2e_fps"] = None

        return {
            "cameras": cameras,
            "spark_cameras": spark_cameras,
            "spark_runtime_sec": spark_time,
            "spark_batches": spark_batches,
        }

    def print_summary(self, prefix: Optional[str] = None):
        summary = self.compute_summary()
        header = prefix or f"Session {self.session_id} metrics"
        divider = "-" * 30
        print(f"\n{divider} {header} {divider}")
        for camera, stats in summary["cameras"].items():
            print(
                f"{camera.upper()}"
                f"mean_e2e_fps={stats.get('mean_e2e_fps') if stats.get('mean_e2e_fps') is not None else 'n/a'}, "
                f"mean_e2e_latency_ms={stats.get('total_mean_e2e_latency_ms') if stats.get('total_mean_e2e_latency_ms') is not None else 'n/a'}"
            )

        spark_cam = summary.get("spark_cameras") or {}
        if spark_cam:
            for camera, stats in spark_cam.items():
                print(
                    f"{camera.upper()} spark: total={stats.get('total_processed')}, "
                    f"mean_e2e_fps={stats.get('mean_e2e_fps') if stats.get('mean_e2e_fps') is not None else 'n/a'}, "
                    f"mean_e2e_latency_ms={stats.get('total_mean_e2e_latency_ms') if stats.get('total_mean_e2e_latency_ms') is not None else 'n/a'}"
                )
        print(divider * 2)


def ensure_kafka_topic(session_id, bootstrap_servers):
    admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    existing_topics = admin_client.list_topics()
    topic_names = [f'cam1_{session_id}', f'cam2_{session_id}' ,f'cam1_processed_{session_id}', f'cam2_processed_{session_id}']
    for topic_name in topic_names:
        if topic_name not in existing_topics:
            try:
                admin_client.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
                print(f"Created Kafka topic: {topic_name}")
            except Exception as e:
                print(f"Error creating topic {topic_name}: {e}")
        else:
            print(f"Kafka topic already exists: {topic_name}")
    admin_client.close()

from streaming.spark_streaming import start_spark
from streaming.kafka_producer import VideoProducer
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(BASE_DIR)
    app.main_loop = asyncio.get_running_loop()
    app.main_loop.create_task(process_websocket_queue())
    yield
    print("[FastAPI] Shutting down...")
    
    for session_id in list(active_sessions.keys()):
        await stop_processing(session_id)
app = FastAPI(title="Vehicle Re-ID Demo", 
              version="1.0.0",
              description="A demo application for Vehicle Re-Identification using FastAPI, Kafka, and Spark Streaming",
              lifespan=lifespan)

app.main_loop = None

    

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for session management
active_sessions: Dict[str, dict] = {}
websocket_connections: Dict[str, WebSocket] = {}

# Build paths relative to the script's location
BASE_DIR = Path(__file__).resolve().parent

(BASE_DIR / "uploads").mkdir(exist_ok=True)
(BASE_DIR.parent / "matching").mkdir(exist_ok=True)
(BASE_DIR / "static").mkdir(exist_ok=True)
(BASE_DIR / "templates").mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/matching", StaticFiles(directory=BASE_DIR.parent / "matching"), name="matching")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"WebSocket connected for session: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"WebSocket disconnected for session: {session_id}")

    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(message)
            except Exception as e:
                print(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)

    async def broadcast_message(self, message: str):
        for session_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error broadcasting to {session_id}: {e}")

manager = ConnectionManager()

async def process_websocket_queue():
    print("[WebSocket Processor] Starting...")
    loop = asyncio.get_running_loop()
    
    processed_count = 0
    last_stats_time = time.time()
    stats_interval = 60  
    
    while True:
        try:
            queue_size = websocket_queue.qsize()
            
            if queue_size > 180: 
                print(f"[WebSocket Processor] WARNING: Queue backlog detected ({queue_size} items). Clearing old frames...")
                session_frames = {}
                discarded = 0
                
                # Process all items in the queue
                while not websocket_queue.empty():
                    try:
                        session_id, message = websocket_queue.get_nowait()
                        if message.get("type") == "frame":
                            key = f"{session_id}_{message.get('camera', 'unknown')}"
                            if key not in session_frames:
                                session_frames[key] = []
                            session_frames[key].append((session_id, message))
                            websocket_queue.task_done()
                            discarded += 1
                        else:
                            websocket_queue.put((session_id, message))
                            websocket_queue.task_done()
                    except Exception:
                        break
                
                for key, frames in session_frames.items():
                    frames.sort(key=lambda x: x[1].get("frame_count", 0), reverse=True)
                    for i, (session_id, message) in enumerate(frames[:2]):
                        websocket_queue.put((session_id, message))
                
                print(f"[WebSocket Processor] Cleared queue backlog. Discarded {discarded} items, kept {sum(len(frames[:2]) for frames in session_frames.values())}.")
            
            try:
                session_id, message = await loop.run_in_executor(
                    None, lambda: websocket_queue.get(timeout=0.5)
                )
                
                if session_id in manager.active_connections:
                    if message.get("type") == "frame":
                        capture_ts_ms = message.get("capture_ts_ms")
                        camera = message.get("camera")
                        session_metrics = active_sessions.get(session_id, {}).get("metrics")
                        if session_metrics and camera:
                            session_metrics.record_frame_sent(camera)
                        if isinstance(capture_ts_ms, (int, float)) and camera:
                            latency_ms = (time.time() * 1000.0) - float(capture_ts_ms)
                            if session_metrics:
                                session_metrics.record_frame_latency(latency_ms, camera)

                    message_json = json.dumps(message)
                    await manager.send_personal_message(message_json, session_id)
                    processed_count += 1
                else:
                    pass
                
                websocket_queue.task_done()
                
                # Free memory
                del message
                
            except asyncio.CancelledError:
                print("[WebSocket Processor] Task cancelled, shutting down...")
                return
            except Exception as e:
                await asyncio.sleep(0.1) 
            
            # Print stats periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                print(f"[WebSocket Processor] Stats: Processed {processed_count} messages in the last {stats_interval} seconds")
                last_stats_time = current_time
                processed_count = 0
                
        except Exception as e:
            print(f"[WebSocket Processor] Error in main loop: {e}")
            await asyncio.sleep(1)

def kafka_consumer_thread(session_id: str, camera: str):
    topic = f'{camera}_processed_{session_id}'
    print(f"[Kafka Consumer {camera.upper()}] Starting consumer thread for session {session_id}")
    
    try:
        print(f"[Kafka Consumer {camera.upper()}] Subscribing to topic: {topic}")
        
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=['localhost:9092'],    
            value_deserializer=None,
            key_deserializer=None,
            fetch_max_bytes=52428800,       
            max_in_flight_requests_per_connection=5,  
            fetch_max_wait_ms=500,          
            fetch_min_bytes=1,              
            consumer_timeout_ms=2000,       
            auto_offset_reset='latest',
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,   
            receive_buffer_bytes=67108864,  
            session_timeout_ms=30000,       
            heartbeat_interval_ms=3000,     
            max_poll_records=10,            
            max_poll_interval_ms=300000,    
            group_id=f'demo_consumer_{camera}_{session_id}_{int(time.time())}'
        )
        
        print(f"[Kafka Consumer {camera.upper()}] Consumer created successfully for session {session_id}")
        
        frame_counter = 0
        last_frame_time = time.time()
        target_fps = 6.0
        frame_interval = 1.0 / target_fps
        print(f"[Kafka Consumer {camera.upper()}] Starting to consume messages at {target_fps} FPS...")
        
        
        consecutive_timeouts = 0
        max_consecutive_timeouts = 100 
        
        while consecutive_timeouts < max_consecutive_timeouts:
            try:
               
                if session_id not in active_sessions:
                    print(f"[Kafka Consumer {camera.upper()}] Session {session_id} no longer active, stopping consumer")
                    break
                
                
                message_batch = consumer.poll(timeout_ms=2000)  
                
                if not message_batch:
                    consecutive_timeouts += 1
                    if consecutive_timeouts % 10 == 0:  
                        print(f"[Kafka Consumer {camera.upper()}] No messages received (timeout {consecutive_timeouts}/{max_consecutive_timeouts})")
                    time.sleep(0.1)  
                    continue
                
                consecutive_timeouts = 0
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        current_time = time.time()
                        elapsed = current_time - last_frame_time
                        if elapsed < frame_interval:
                            continue  

                        last_frame_time = current_time
                        fps_sample = (1.0 / elapsed) if elapsed > 0 else None
                        
                        try:
                            capture_ts_ms = None
                            try:
                                key_bytes = getattr(message, "key", None)
                                if key_bytes:
                                    key_str = key_bytes.decode("utf-8", errors="ignore")
                                    ts_part = key_str.split(":", 1)[0]
                                    capture_ts_ms = int(ts_part)
                            except Exception:
                                capture_ts_ms = None

                            frame_data = message.value
                            
                            if frame_counter % 100 == 0:
                                print(f"[Kafka Consumer {camera.upper()}] Processing frame {frame_counter}, data size: {len(frame_data)} bytes")
                            
                            frame_buffer = np.frombuffer(frame_data, dtype=np.uint8)
                            final_img = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                            
                            if final_img is None:
                                print(f"[Kafka Consumer {camera.upper()}] ERROR: Could not decode image, skipping...")
                                continue

                            _, buffer = cv2.imencode('.jpg', final_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            websocket_message = {
                                "type": "frame",
                                "camera": camera,
                                "data": frame_b64,
                                "timestamp": datetime.now().isoformat(),
                                "frame_count": frame_counter + 1,
                                "capture_ts_ms": capture_ts_ms,
                            }
                            
                            websocket_queue.put((session_id, websocket_message))

                            del frame_buffer, final_img, buffer, frame_b64
                            frame_counter += 1
                            
                            if frame_counter % 50 == 0:
                                import gc
                                gc.collect()
                                
                            session_metrics = active_sessions.get(session_id, {}).get("metrics")
                            if session_metrics:
                                if fps_sample:
                                    session_metrics.record_frontend_fps(camera, fps_sample)

                        except Exception as e:
                            print(f"[Kafka Consumer {camera.upper()}] Error processing message: {e}")
                            continue
                            
            except Exception as e:
                print(f"[Kafka Consumer {camera.upper()}] Error in consumer loop: {e}")
                consecutive_timeouts += 1
                time.sleep(1)

        print(f"[Kafka Consumer {camera.upper()}] Consumer loop ended")
                
    except Exception as e:
        print(f"[Kafka Consumer {camera.upper()}] Error in consumer thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            consumer.close()
            print(f"[Kafka Consumer {camera.upper()}] Consumer closed for session {session_id}")
        except:
            pass

def matching_monitor_thread(session_id: str):
    matching_dir = BASE_DIR.parent / "matching"
    processed_files = set()
    
    print(f"[Matching Monitor] Started for session {session_id}")
    
    while session_id in active_sessions:
        try:
            for file_path in matching_dir.glob("*.jpg"):
                if file_path.name not in processed_files:
                    processed_files.add(file_path.name)
                    
                    websocket_message = {
                        "type": "match",
                        "filename": file_path.name,
                        "path": f"/matching/{file_path.name}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    websocket_queue.put((session_id, websocket_message))
                    print(f"[Matching Monitor] Match found and queued: {file_path.name}")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"[Matching Monitor] Error: {e}")
            time.sleep(5)
    
    print(f"[Matching Monitor] Stopped for session {session_id}")

@app.get("/latest_matches")
async def get_latest_matches():
    matching_dir = BASE_DIR.parent / "matching"
    
    try:
        image_files = []
        
        print(f"[Latest Matches] Looking for images in {matching_dir}")
        
        flat_files = list(matching_dir.glob("*.jpg"))
        image_files.extend(flat_files)
        print(f"[Latest Matches] Found {len(flat_files)} flat files")
        
        for subdir in matching_dir.iterdir():
            if subdir.is_dir():
                subdir_files = list(subdir.glob("*.jpg"))
                print(f"[Latest Matches] Found {len(subdir_files)} files in subdirectory {subdir.name}")
                image_files.extend(subdir_files)
        
        print(f"[Latest Matches] Total images found: {len(image_files)}")
        
        if not image_files:
            test_dir = matching_dir / "test1"
            if test_dir.exists():
                test_images = list(test_dir.glob("*.jpg"))
                if test_images:
                    print(f"[Latest Matches] Found {len(test_images)} test images in {test_dir}")
                    image_files.extend(test_images)
        
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        matches = []
        for img_path in image_files[:20]:
            
            relative_path = img_path.relative_to(matching_dir)
            
            camera_tag = "cam1" if "cam1" in img_path.name.lower() else "cam2" if "cam2" in img_path.name.lower() else "cam1" if "test1" in str(img_path) else "cam2"
            
            matches.append({
                "filename": f"{camera_tag}_{img_path.name}",  # Add camera prefix if not present
                "path": f"/matching/{relative_path.as_posix()}",
                "thumbnail_url": f"/matching_resized/{img_path.name}?width=90&height=120",
                "small_thumbnail_url": f"/matching_resized/{img_path.name}?width=60&height=80",
                "timestamp": datetime.fromtimestamp(img_path.stat().st_mtime).isoformat()
            })
        
        print(f"[Latest Matches] Returning {len(matches)} matches")
        return {"matches": matches}
        
    except Exception as e:
        print(f"Error getting latest matches: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Could not retrieve latest matches.")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    template_path = BASE_DIR / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/start_session")
async def start_session():
    session_id = str(uuid.uuid4())[:8]
    
    session_data = {
        "id": session_id,
        "status": "initialized",
        "created_at": datetime.now().isoformat(),
        "cameras": {"cam1": None, "cam2": None},
        "spark_thread": None,
        "kafka_threads": {"cam1": None, "cam2": None},
        "matching_thread": None,
        "metrics": SessionMetrics(session_id),
    }
    
    active_sessions[session_id] = session_data
    
    print(f"[Session] Created new session: {session_id}")
    
    return {"session_id": session_id, "status": "initialized"}

@app.post("/upload_video/{session_id}")
async def upload_video(session_id: str, camera: str, file: UploadFile = File(...)):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if camera not in ["cam1", "cam2"]:
        raise HTTPException(status_code=400, detail="Camera must be 'cam1' or 'cam2'")
    
    # Save uploaded file
    upload_dir = BASE_DIR / "uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / f"{camera}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Update session data
    active_sessions[session_id]["cameras"][camera] = str(file_path)
    
    print(f"[Upload] Video uploaded for {camera} in session {session_id}: {file_path}")
    
    return {"message": f"Video uploaded for {camera}", "file_path": str(file_path)}

@app.post("/start_processing/{session_id}")
async def start_processing(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[session_id]
    
    # Check if both videos are uploaded
    if not session_data["cameras"]["cam1"] or not session_data["cameras"]["cam2"]:
        raise HTTPException(status_code=400, detail="Both camera videos must be uploaded")

    try:
        print(f"[Processing] Creating Kafka topics for session {session_id}")
        ensure_kafka_topic(session_id, bootstrap_servers='localhost:9092')
        
        print(f"[Processing] Starting Spark streaming for session {session_id}")
        spark_base_dir = BASE_DIR.parent 
        session_data["metrics"].start_spark_run()
        spark_thread = start_spark(
            session_id,
            base_dir=spark_base_dir,
            progress_callback=session_data["metrics"].record_micro_batch,
            session_metrics=session_data["metrics"],
        )
        session_data["spark_thread"] = spark_thread
        
        print(f"[Processing] Waiting for Spark to initialize...")
        await asyncio.sleep(30)  
        
        # Start video producers
        print(f"[Processing] Starting video producers for session {session_id}")

        producer1 = VideoProducer(
            topic=f"cam1_{session_id}",
            bootstrap_servers="localhost:9092",
            fps=6.0, 
            mode='streaming'
        )
        
        producer2 = VideoProducer(
            topic=f"cam2_{session_id}",
            bootstrap_servers="localhost:9092",
            fps=6.0,  # Set to 6 FPS for controlled streaming
            mode='streaming'
        )
        
        producer1_thread = threading.Thread(target=producer1.start_streaming, args=(session_data["cameras"]["cam1"],))
        producer2_thread = threading.Thread(target=producer2.start_streaming, args=(session_data["cameras"]["cam2"],))
        
        producer1_thread.start()
        producer2_thread.start()
        
        print(f"[Processing] Starting Kafka consumer threads for session {session_id}")
        
        kafka_thread_cam1 = threading.Thread(target=kafka_consumer_thread, args=(session_id, "cam1"))
        kafka_thread_cam1.start()
        session_data["kafka_threads"]["cam1"] = kafka_thread_cam1
        
        kafka_thread_cam2 = threading.Thread(target=kafka_consumer_thread, args=(session_id, "cam2"))
        kafka_thread_cam2.start()
        session_data["kafka_threads"]["cam2"] = kafka_thread_cam2
        
        matching_thread = threading.Thread(target=matching_monitor_thread, args=(session_id,))
        matching_thread.start()
        session_data["matching_thread"] = matching_thread
        
        session_data["status"] = "processing"
        session_data["processing_start_time"] = datetime.now().isoformat()
        
        print(f"[Processing] Started processing for session {session_id}")
        
        return {
            "message": "Processing started", 
            "session_id": session_id,
            "start_time": session_data["processing_start_time"]
        }
        
    except Exception as e:
        print(f"[Processing] Error starting processing: {e}")
        session_data["status"] = "error"
        metrics = session_data.get("metrics")
        if metrics:
            metrics.stop_spark_run()
            metrics.print_summary(prefix=f"Session {session_id} metrics before failure")
        raise HTTPException(status_code=500, detail=f"Error starting processing: {str(e)}")

@app.post("/stop_processing/{session_id}")
async def stop_processing(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[session_id]
    session_data["status"] = "stopping"
    metrics = session_data.get("metrics")
    if metrics:
        metrics.stop_spark_run()
        metrics.print_summary(prefix=f"Session {session_id} final metrics")
    
    if session_id in active_sessions:
        del active_sessions[session_id]
    
    print(f"[Processing] Stopped processing for session {session_id}")
    
    return {"message": "Processing stopped", "session_id": session_id}

@app.get("/session_status/{session_id}")
async def get_session_status(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[session_id]
    
    spark_thread_alive = False
    kafka_threads_alive = {"cam1": False, "cam2": False}
    matching_thread_alive = False
    
    if session_data["spark_thread"]:
        spark_thread_alive = session_data["spark_thread"].is_alive()
        
    if session_data["kafka_threads"]["cam1"]:
        kafka_threads_alive["cam1"] = session_data["kafka_threads"]["cam1"].is_alive()
        
    if session_data["kafka_threads"]["cam2"]:
        kafka_threads_alive["cam2"] = session_data["kafka_threads"]["cam2"].is_alive()
        
    if session_data["matching_thread"]:
        matching_thread_alive = session_data["matching_thread"].is_alive()
    
    return {
        "session_id": session_id,
        "status": session_data["status"],
        "created_at": session_data["created_at"],
        "cameras": {
            "cam1": session_data["cameras"]["cam1"] is not None,
            "cam2": session_data["cameras"]["cam2"] is not None
        },
        "threads": {
            "spark_alive": spark_thread_alive,
            "kafka_cam1_alive": kafka_threads_alive["cam1"],
            "kafka_cam2_alive": kafka_threads_alive["cam2"],
            "matching_alive": matching_thread_alive
        },
        "processing_info": {
            "start_time": session_data.get("processing_start_time", None),
            "elapsed_time": str(datetime.now() - datetime.fromisoformat(session_data.get("processing_start_time", datetime.now().isoformat()))) if session_data.get("processing_start_time") else "Not started"
        }
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    heartbeat_interval = 30  
    last_heartbeat = time.time()
    
    try:
        while True:
            await asyncio.sleep(0.5)
     
            current_time = time.time()
            if current_time - last_heartbeat >= heartbeat_interval:
                try:
                    await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
                    last_heartbeat = current_time
                except Exception as e:
                    print(f"[WebSocket] Error sending heartbeat: {e}")
                    break
                    
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for session {session_id}")
        manager.disconnect(session_id)
    except Exception as e:
        print(f"[WebSocket] Error in WebSocket connection: {e}")
        manager.disconnect(session_id)
        print(f"WebSocket disconnected and removed for session: {session_id}")

@app.get("/matching_files")
async def get_matching_files():
    matching_dir = BASE_DIR.parent / "matching"
    files = []
    
    if matching_dir.exists():
        for file_path in matching_dir.glob("*.jpg"):
            files.append({
                "filename": file_path.name,
                "path": f"/matching/{file_path.name}",
                "created_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    return {"files": files}

@app.get("/debug/{session_id}")
async def debug_session(session_id: str):
    debug_info = {
        "session_id": session_id,
        "session_exists": session_id in active_sessions,
        "websocket_connected": session_id in manager.active_connections,
        "kafka_topics": [],
        "database_path": f"./chroma_vehicle_reid_streaming/{session_id}",
        "database_exists": False,
        "matching_files": 0
    }
    
    if session_id in active_sessions:
        session_data = active_sessions[session_id]
        debug_info["session_status"] = session_data["status"]
        debug_info["cameras_uploaded"] = {
            "cam1": session_data["cameras"]["cam1"] is not None,
            "cam2": session_data["cameras"]["cam2"] is not None
        }
        debug_info["threads"] = {
            "spark_thread": session_data["spark_thread"] is not None,
            "kafka_thread_cam1": session_data["kafka_threads"]["cam1"] is not None,
            "kafka_thread_cam2": session_data["kafka_threads"]["cam2"] is not None,
            "matching_thread": session_data["matching_thread"] is not None
        }
    
    try:
        from kafka.admin import KafkaAdminClient
        admin_client = KafkaAdminClient(bootstrap_servers=['localhost:9092'])
        all_topics = admin_client.list_topics()
        
        expected_topics = [
            f'cam1_{session_id}',
            f'cam2_{session_id}',
            f'cam1_processed_{session_id}',
            f'cam2_processed_{session_id}'
        ]
        
        for topic in expected_topics:
            debug_info["kafka_topics"].append({
                "topic": topic,
                "exists": topic in all_topics
            })
        
        admin_client.close()
    except Exception as e:
        debug_info["kafka_error"] = str(e)
    
    # Check database
    import os
    db_path = f"./chroma_vehicle_reid_streaming/{session_id}"
    debug_info["database_exists"] = os.path.exists(db_path)
    
    if debug_info["database_exists"]:
        try:
            db_files = os.listdir(db_path)
            debug_info["database_files"] = db_files
        except Exception as e:
            debug_info["database_error"] = str(e)
    
    matching_dir = BASE_DIR.parent / "matching"
    if matching_dir.exists():
        debug_info["matching_files"] = len(list(matching_dir.glob("*.jpg")))
    
    return debug_info

@app.get("/debug_page", response_class=HTMLResponse)
async def debug_page():
    debug_template_path = BASE_DIR / "debug.html"
    with open(debug_template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
    

@app.post("/test_websocket/{session_id}")
async def test_websocket(session_id: str):
    if session_id not in manager.active_connections:
        return {"error": "WebSocket not connected for this session"}
    
    test_message = {
        "type": "test",
        "message": "WebSocket connection test",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        await manager.send_personal_message(
            json.dumps(test_message), 
            session_id
        )
        return {"success": True, "message": "Test message sent"}
    except Exception as e:
        return {"error": f"Failed to send test message: {str(e)}"}

@app.post("/test_frame/{session_id}")
async def test_frame(session_id: str, camera: str = "cam1"):
    if session_id not in manager.active_connections:
        return {"error": "WebSocket not connected for this session"}
    
    if camera not in ["cam1", "cam2"]:
        return {"error": "Camera must be 'cam1' or 'cam2'"}
    
    try:
        import numpy as np
        import cv2
        import base64
        
        height, width = 480, 640
        test_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if camera == "cam1":
            test_img[:, :] = [255, 0, 0]
        else:
            test_img[:, :] = [0, 255, 0]  
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(test_img, f"{camera.upper()} - {timestamp}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', test_img)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        websocket_message = {
            "type": "frame",
            "camera": camera,
            "data": frame_b64,
            "timestamp": datetime.now().isoformat(),
            "frame_count": 999,
            "frame_hash": "test"
        }
        
        await manager.send_personal_message(
            json.dumps(websocket_message), 
            session_id
        )
        
        return {"success": True, "message": f"Test frame sent to {camera}"}
        
    except Exception as e:
        return {"error": f"Failed to send test frame: {str(e)}"}
@app.get("/matching_resized/{filename}")
async def get_resized_matching_image(filename: str, width: int = 90, height: int = 120):
    try:
        matching_dir = BASE_DIR.parent / "matching"
        
        clean_filename = filename
        if clean_filename.startswith(("cam1_", "cam2_")):
            clean_filename = clean_filename[5:]  # Remove the prefix
        
        print(f"[Resize] Looking for image: {clean_filename}, original request: {filename}")
        
        # Try to find the file in the matching directory or subdirectories
        image_path = None
        
        # First check if it's a flat file
        flat_file = matching_dir / clean_filename
        if flat_file.exists():
            image_path = flat_file
            print(f"[Resize] Found image as flat file: {flat_file}")
        else:
            # Check subdirectories
            for subdir in matching_dir.iterdir():
                if subdir.is_dir():
                    subdir_file = subdir / clean_filename
                    if subdir_file.exists():
                        image_path = subdir_file
                        print(f"[Resize] Found image in subdirectory: {subdir_file}")
                        break
        
        if not image_path:
            print(f"[Resize] Image not found: {clean_filename}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Read and resize the image
        import cv2
        import numpy as np
        from io import BytesIO
        
        # Read the original image
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            raise HTTPException(status_code=500, detail="Could not read image")
        
        # Create consistent thumbnail dimensions with padding to preserve content
        h, w = original_img.shape[:2]
        aspect_ratio = w / h
        target_ratio = width / height
        
        # Calculate the size to fit the image within the target dimensions
        if aspect_ratio > target_ratio:
            # Image is wider, fit to width
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            # Image is taller, fit to height
            new_height = height
            new_width = int(height * aspect_ratio)
        
        # Resize the image maintaining aspect ratio
        resized_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create a canvas with the target dimensions and fill with neutral color
        canvas = np.full((height, width, 3), 240, dtype=np.uint8)  # Light gray background
        
        # Calculate position to center the resized image
        y_offset = (height - new_height) // 2
        x_offset = (width - new_width) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
        
        resized_img = canvas
        
        # Encode image to JPEG with higher quality for small thumbnails
        quality = 95 if max(width, height) <= 150 else 85
        _, buffer = cv2.imencode('.jpg', resized_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # Return the resized image
        from fastapi.responses import Response
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}  # Cache for 1 hour
        )
        
    except Exception as e:
        print(f"Error serving resized image {filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not process image")


@app.get("/test_resize")
async def test_resize():
    try:
        matching_dir = BASE_DIR.parent / "matching"
        
        # Get the first available image
        test_image = None
        for img_path in matching_dir.glob("*.jpg"):
            test_image = img_path
            break
        
        if not test_image:
            # Check subdirectories
            for subdir in matching_dir.iterdir():
                if subdir.is_dir():
                    for img_path in subdir.glob("*.jpg"):
                        test_image = img_path
                        break
                if test_image:
                    break
        
        if test_image:
            return {
                "success": True,
                "test_image": test_image.name,
                "original_url": f"/matching/{test_image.relative_to(matching_dir).as_posix()}",
                "thumbnail_url": f"/matching_resized/{test_image.name}?width=300&height=200",
                "small_thumbnail_url": f"/matching_resized/{test_image.name}?width=150&height=100"
            }
        else:
            return {"success": False, "message": "No test images found"}
            
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    import os

    try:
        uvicorn.run(app, host="localhost", port=8000)
    except KeyboardInterrupt:
        print("\n[FastAPI] KeyboardInterrupt received. Printing session metrics before exit...")
        for session_id, session in list(active_sessions.items()):
            metrics = session.get("metrics")
            if metrics:
                metrics.stop_spark_run()
                metrics.print_summary(prefix=f"Session {session_id} final metrics")
        print("[FastAPI] Shutdown complete.")