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
from typing import Dict, List
import shutil
from pathlib import Path
import base64
import tempfile
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from kafka.admin import KafkaAdminClient, NewTopic
import time
from queue import Queue
import numpy as np

# Global queue for WebSocket messages
websocket_queue = Queue()

def ensure_kafka_topic(session_id, bootstrap_servers):
    """Check if a Kafka topic exists, and create it if not."""
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
            # Return a consumer for the topic
    admin_client.close()

# Add the parent directory to Python path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming.spark_services.spark_streaming import start_spark
from streaming.kafka_services.producer import VideoProducer
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Capture the main event loop on startup."""
    print(BASE_DIR)
    app.main_loop = asyncio.get_running_loop()
    # Start a background task to process the WebSocket message queue
    app.main_loop.create_task(process_websocket_queue())
    yield
    """Shutdown event handler"""
    print("[FastAPI] Shutting down...")
    
    # Clean up all active sessions
    for session_id in list(active_sessions.keys()):
        await stop_processing(session_id)
app = FastAPI(title="Vehicle Re-ID Demo", 
              version="1.0.0",
              description="A demo application for Vehicle Re-Identification using FastAPI, Kafka, and Spark Streaming",
              lifespan=lifespan)

# Store the main event loop
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

# Create necessary directories
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
    """Process messages from the global queue and send them to the correct WebSocket.
    Optimized for memory efficiency and to handle queue backlog situations."""
    print("[WebSocket Processor] Starting...")
    loop = asyncio.get_running_loop()
    
    # Track processing statistics
    processed_count = 0
    last_stats_time = time.time()
    stats_interval = 60  # Report stats every minute
    
    while True:
        try:
            # Get current queue size for monitoring
            queue_size = websocket_queue.qsize()
            
            # Handle backlog situation - drop old frames if queue gets too big
            if queue_size > 180:  # If more than 30 frames are queued (5 seconds at 6 FPS)
                print(f"[WebSocket Processor] WARNING: Queue backlog detected ({queue_size} items). Clearing old frames...")
                # Keep only the most recent 5 frames for each session
                session_frames = {}
                discarded = 0
                
                # Process all items in the queue
                while not websocket_queue.empty():
                    try:
                        session_id, message = websocket_queue.get_nowait()
                        if message.get("type") == "frame":
                            # Keep only most recent frames per session/camera
                            key = f"{session_id}_{message.get('camera', 'unknown')}"
                            if key not in session_frames:
                                session_frames[key] = []
                            session_frames[key].append((session_id, message))
                            websocket_queue.task_done()
                            discarded += 1
                        else:
                            # Re-add non-frame messages
                            websocket_queue.put((session_id, message))
                            websocket_queue.task_done()
                    except Exception:
                        break
                
                # Re-add only the 2 most recent frames per session/camera
                for key, frames in session_frames.items():
                    # Sort by frame count if available
                    frames.sort(key=lambda x: x[1].get("frame_count", 0), reverse=True)
                    # Add back the 2 most recent frames
                    for i, (session_id, message) in enumerate(frames[:2]):
                        websocket_queue.put((session_id, message))
                
                print(f"[WebSocket Processor] Cleared queue backlog. Discarded {discarded} items, kept {sum(len(frames[:2]) for frames in session_frames.values())}.")
            
            # Normal processing - get message from the queue
            try:
                # Use a timeout to prevent blocking indefinitely
                session_id, message = await loop.run_in_executor(
                    None, lambda: websocket_queue.get(timeout=0.5)
                )
                
                # Send the message if the connection is still active
                if session_id in manager.active_connections:
                    # Convert to JSON string once
                    message_json = json.dumps(message)
                    await manager.send_personal_message(message_json, session_id)
                    processed_count += 1
                else:
                    # Connection not active, silently discard
                    pass
                
                # Mark as done
                websocket_queue.task_done()
                
                # Free memory
                del message
                
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                print("[WebSocket Processor] Task cancelled, shutting down...")
                return
            except Exception as e:
                # Could be queue.Empty or other exception
                await asyncio.sleep(0.1)  # Short sleep if no messages
            
            # Print stats periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                print(f"[WebSocket Processor] Stats: Processed {processed_count} messages in the last {stats_interval} seconds")
                last_stats_time = current_time
                processed_count = 0
                
        except Exception as e:
            print(f"[WebSocket Processor] Error in main loop: {e}")
            # In case of an error, sleep a bit before retrying
            await asyncio.sleep(1)

def kafka_consumer_thread(session_id: str, camera: str):
    """Thread function to consume processed frames from a specific camera"""
    topic = f'{camera}_processed_{session_id}'
    print(f"[Kafka Consumer {camera.upper()}] Starting consumer thread for session {session_id}")
    
    try:
        print(f"[Kafka Consumer {camera.upper()}] Subscribing to topic: {topic}")
        
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=['localhost:9092'],    
            value_deserializer=None,
            key_deserializer=None,
            fetch_max_bytes=52428800,       # 50MB
            max_in_flight_requests_per_connection=5,  # Reduced from 10
            fetch_max_wait_ms=500,          # Further reduced for responsiveness
            fetch_min_bytes=1,              
            consumer_timeout_ms=2000,       # Increased timeout
            auto_offset_reset='latest',
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,   # Auto-commit every second
            receive_buffer_bytes=67108864,  # 64MB
            session_timeout_ms=30000,       # Increased session timeout
            heartbeat_interval_ms=3000,     # Heartbeat every 3 seconds
            max_poll_records=10,            # Limit records per poll
            max_poll_interval_ms=300000,    # 5 minutes max poll interval
            group_id=f'demo_consumer_{camera}_{session_id}_{int(time.time())}'
        )
        
        print(f"[Kafka Consumer {camera.upper()}] Consumer created successfully for session {session_id}")
        
        frame_counter = 0
        last_frame_time = time.time()
        target_fps = 6.0
        frame_interval = 1.0 / target_fps
        print(f"[Kafka Consumer {camera.upper()}] Starting to consume messages at {target_fps} FPS...")
        
        # Improved timeout handling
        consecutive_timeouts = 0
        max_consecutive_timeouts = 100  # Reduced from 10000
        
        while consecutive_timeouts < max_consecutive_timeouts:
            try:
                # Check if session is still active first
                if session_id not in active_sessions:
                    print(f"[Kafka Consumer {camera.upper()}] Session {session_id} no longer active, stopping consumer")
                    break
                
                # Get message with better timeout handling
                message_batch = consumer.poll(timeout_ms=2000)  # Increased timeout
                
                if not message_batch:
                    consecutive_timeouts += 1
                    if consecutive_timeouts % 10 == 0:  # Log every 10 timeouts
                        print(f"[Kafka Consumer {camera.upper()}] No messages received (timeout {consecutive_timeouts}/{max_consecutive_timeouts})")
                    time.sleep(0.1)  # Slightly longer sleep
                    continue
                
                # Reset timeout counter on successful message
                consecutive_timeouts = 0
                
                # Process messages with frame rate control
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        # Frame rate control
                        current_time = time.time()
                        if current_time - last_frame_time < frame_interval:
                            continue  # Skip this frame to maintain target FPS
                        
                        last_frame_time = current_time
                        
                        try:
                            # Decode the frame
                            frame_data = message.value
                            
                            # Only log every 100th frame to reduce console spam
                            if frame_counter % 100 == 0:
                                print(f"[Kafka Consumer {camera.upper()}] Processing frame {frame_counter}, data size: {len(frame_data)} bytes")
                            
                            # Decode image from bytes
                            frame_buffer = np.frombuffer(frame_data, dtype=np.uint8)
                            final_img = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                            
                            if final_img is None:
                                print(f"[Kafka Consumer {camera.upper()}] ERROR: Could not decode image, skipping...")
                                continue

                            # Encode frame to JPEG for web transmission
                            _, buffer = cv2.imencode('.jpg', final_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Create the message dictionary
                            websocket_message = {
                                "type": "frame",
                                "camera": camera,
                                "data": frame_b64,
                                "timestamp": datetime.now().isoformat(),
                                "frame_count": frame_counter + 1
                            }
                            
                            # Send to WebSocket via the main event loop
                            websocket_queue.put((session_id, websocket_message))
                            
                            # Clean up memory immediately
                            del frame_buffer, final_img, buffer, frame_b64
                            frame_counter += 1
                            
                            # More frequent garbage collection
                            if frame_counter % 50 == 0:
                                import gc
                                gc.collect()
                                
                        except Exception as e:
                            print(f"[Kafka Consumer {camera.upper()}] Error processing message: {e}")
                            continue
                            
            except Exception as e:
                print(f"[Kafka Consumer {camera.upper()}] Error in consumer loop: {e}")
                consecutive_timeouts += 1
                time.sleep(1)  # Wait before retrying
                    
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
    """Thread function to monitor matching folder for new vehicle matches"""
    matching_dir = BASE_DIR.parent / "matching"
    processed_files = set()
    
    print(f"[Matching Monitor] Started for session {session_id}")
    
    while session_id in active_sessions:
        try:
            # Check for new matching files
            for file_path in matching_dir.glob("*.jpg"):
                if file_path.name not in processed_files:
                    processed_files.add(file_path.name)
                    
                    # Send matching notification
                    websocket_message = {
                        "type": "match",
                        "filename": file_path.name,
                        "path": f"/matching/{file_path.name}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Put the message in the queue to be sent by the main loop
                    websocket_queue.put((session_id, websocket_message))
                    print(f"[Matching Monitor] Match found and queued: {file_path.name}")
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"[Matching Monitor] Error: {e}")
            time.sleep(5)
    
    print(f"[Matching Monitor] Stopped for session {session_id}")

@app.get("/latest_matches")
async def get_latest_matches():
    """Get the latest cross-camera matching images from the filesystem."""
    matching_dir = BASE_DIR.parent / "matching"
    
    try:
        image_files = []
        
        # Debug log for troubleshooting
        print(f"[Latest Matches] Looking for images in {matching_dir}")
        
        # First, get all image files directly from the matching folder (flat structure)
        flat_files = list(matching_dir.glob("*.jpg"))
        image_files.extend(flat_files)
        print(f"[Latest Matches] Found {len(flat_files)} flat files")
        
        # Then, get all image files from subdirectories (legacy structure)
        for subdir in matching_dir.iterdir():
            if subdir.is_dir():
                subdir_files = list(subdir.glob("*.jpg"))
                print(f"[Latest Matches] Found {len(subdir_files)} files in subdirectory {subdir.name}")
                image_files.extend(subdir_files)
        
        print(f"[Latest Matches] Total images found: {len(image_files)}")
        
        if not image_files:
            # For testing: use some hardcoded test images if none are found
            test_dir = matching_dir / "test1"
            if test_dir.exists():
                test_images = list(test_dir.glob("*.jpg"))
                if test_images:
                    print(f"[Latest Matches] Found {len(test_images)} test images in {test_dir}")
                    image_files.extend(test_images)
        
        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        matches = []
        # Limit to the latest 20 matches
        for img_path in image_files[:20]:
            
            # Calculate the relative path from the matching directory
            relative_path = img_path.relative_to(matching_dir)
            
            # Create camera tag based on filename or path
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
    """Serve the main HTML page"""
    template_path = BASE_DIR / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/start_session")
async def start_session():
    """Start a new processing session"""
    session_id = str(uuid.uuid4())[:8]
    
    # Create session data
    session_data = {
        "id": session_id,
        "status": "initialized",
        "created_at": datetime.now().isoformat(),
        "cameras": {"cam1": None, "cam2": None},
        "spark_thread": None,
        "kafka_threads": {"cam1": None, "cam2": None},
        "matching_thread": None
    }
    
    active_sessions[session_id] = session_data
    
    print(f"[Session] Created new session: {session_id}")
    
    return {"session_id": session_id, "status": "initialized"}

@app.post("/upload_video/{session_id}")
async def upload_video(session_id: str, camera: str, file: UploadFile = File(...)):
    """Upload video for a specific camera"""
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
    """Start processing the uploaded videos"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[session_id]
    
    # Check if both videos are uploaded
    if not session_data["cameras"]["cam1"] or not session_data["cameras"]["cam2"]:
        raise HTTPException(status_code=400, detail="Both camera videos must be uploaded")
    
    try:
        # Create Kafka topics first
        print(f"[Processing] Creating Kafka topics for session {session_id}")
        ensure_kafka_topic(session_id, bootstrap_servers='localhost:9092')
        
        # Start Spark streaming, passing the absolute base directory
        print(f"[Processing] Starting Spark streaming for session {session_id}")
        # The base directory for spark is the root of the Vehicle-Re-ID project
        spark_base_dir = BASE_DIR.parent 
        spark_thread = start_spark(session_id, base_dir=spark_base_dir)
        session_data["spark_thread"] = spark_thread
        
        # Wait for Spark to initialize
        print(f"[Processing] Waiting for Spark to initialize...")
        await asyncio.sleep(30)  # Increased to 30 seconds to ensure Spark is ready
        
        # Start video producers
        print(f"[Processing] Starting video producers for session {session_id}")
        
        # Start cam1 producer with controlled frame rate
        producer1 = VideoProducer(
            video_path=session_data["cameras"]["cam1"],
            topic_name=f"cam1_{session_id}",
            bootstrap_servers="localhost:9092",
            fps= None  # Set to 6 FPS for controlled streaming
        )
        
        # Start cam2 producer with controlled frame rate
        producer2 = VideoProducer(
            video_path=session_data["cameras"]["cam2"],
            topic_name=f"cam2_{session_id}",
            bootstrap_servers="localhost:9092",
            fps= None  # Set to 6 FPS for controlled streaming
        )
        
        # Start producers in threads
        producer1_thread = threading.Thread(target=producer1.start_streaming)
        producer2_thread = threading.Thread(target=producer2.start_streaming)
        
        producer1_thread.start()
        producer2_thread.start()
        
        # Start Kafka consumer threads for each camera
        print(f"[Processing] Starting Kafka consumer threads for session {session_id}")
        
        # Consumer for cam1
        kafka_thread_cam1 = threading.Thread(target=kafka_consumer_thread, args=(session_id, "cam1"))
        kafka_thread_cam1.start()
        session_data["kafka_threads"]["cam1"] = kafka_thread_cam1
        
        # Consumer for cam2
        kafka_thread_cam2 = threading.Thread(target=kafka_consumer_thread, args=(session_id, "cam2"))
        kafka_thread_cam2.start()
        session_data["kafka_threads"]["cam2"] = kafka_thread_cam2
        
        # Start matching monitor thread
        matching_thread = threading.Thread(target=matching_monitor_thread, args=(session_id,))
        matching_thread.start()
        session_data["matching_thread"] = matching_thread
        
        # Update session status
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
        raise HTTPException(status_code=500, detail=f"Error starting processing: {str(e)}")

@app.post("/stop_processing/{session_id}")
async def stop_processing(session_id: str):
    """Stop processing for a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[session_id]
    session_data["status"] = "stopping"
    
    # Clean up session
    if session_id in active_sessions:
        del active_sessions[session_id]
    
    print(f"[Processing] Stopped processing for session {session_id}")
    
    return {"message": "Processing stopped", "session_id": session_id}

@app.get("/session_status/{session_id}")
async def get_session_status(session_id: str):
    """Get the status of a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[session_id]
    
    # Get additional status information
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
    """WebSocket endpoint for real-time frame streaming with heartbeat mechanism"""
    await manager.connect(websocket, session_id)
    
    heartbeat_interval = 30  # Send a heartbeat every 30 seconds to keep connection alive
    last_heartbeat = time.time()
    
    try:
        while True:
            # Use a short sleep to keep the connection responsive
            await asyncio.sleep(0.5)
            
            # Send a lightweight heartbeat message periodically
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
    """Get list of matching files"""
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
    """Debug endpoint to check session status and components"""
    debug_info = {
        "session_id": session_id,
        "session_exists": session_id in active_sessions,
        "websocket_connected": session_id in manager.active_connections,
        "kafka_topics": [],
        "database_path": f"./chroma_vehicle_reid_streaming/{session_id}",
        "database_exists": False,
        "matching_files": 0
    }
    
    # Check session data
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
    
    # Check Kafka topics
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
    
    # Check matching files
    matching_dir = BASE_DIR.parent / "matching"
    if matching_dir.exists():
        debug_info["matching_files"] = len(list(matching_dir.glob("*.jpg")))
    
    return debug_info

@app.get("/debug_page", response_class=HTMLResponse)
async def debug_page():
    """Serve the debug HTML page"""
    debug_template_path = BASE_DIR / "debug.html"
    with open(debug_template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
    

@app.post("/test_websocket/{session_id}")
async def test_websocket(session_id: str):
    """Test WebSocket connection by sending a test message"""
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
    """Send a test frame to WebSocket for debugging"""
    if session_id not in manager.active_connections:
        return {"error": "WebSocket not connected for this session"}
    
    if camera not in ["cam1", "cam2"]:
        return {"error": "Camera must be 'cam1' or 'cam2'"}
    
    try:
        # Create a simple test image
        import numpy as np
        import cv2
        import base64
        
        # Create a colored test image with timestamp
        height, width = 480, 640
        test_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Different colors for different cameras
        if camera == "cam1":
            test_img[:, :] = [255, 0, 0]  # Red
        else:
            test_img[:, :] = [0, 255, 0]  # Green
        
        # Add timestamp text
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(test_img, f"{camera.upper()} - {timestamp}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', test_img)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send test frame
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
    """Serve a resized version of a matching image"""
    try:
        matching_dir = BASE_DIR.parent / "matching"
        
        # Clean filename by removing any camera prefix (cam1_, cam2_)
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
    """Test endpoint to verify image resizing works"""
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
    uvicorn.run(app, host="localhost", port=8000)
