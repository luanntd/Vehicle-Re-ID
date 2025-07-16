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
import time
from queue import Queue
import numpy as np

# Global queue for WebSocket messages
websocket_queue = Queue()

# Add the parent directory to Python path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import direct pipeline components (like run_reid.py)
from realtime_reid.vehicle_detector import VehicleDetector
from realtime_reid.feature_extraction import VehicleDescriptor
from realtime_reid.classifier_chromadb import ChromaDBVehicleReID
from realtime_reid.pipeline import Pipeline
import gc
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

def direct_video_processing_thread(session_id: str):
    """Process both videos directly using the run_reid pipeline approach"""
    print(f"[Direct Processing] Starting for session {session_id}")
    
    if session_id not in active_sessions:
        print(f"[Direct Processing] Session {session_id} not found")
        return
    
    session_data = active_sessions[session_id]
    
    try:
        # Initialize components with ChromaDB (same as run_reid)
        # Look for model files in the parent directory
        detector_path = 'best_20.pt'
        descriptor_path = 'best_osnet_model.pth'
        
        # Try parent directory if not found in current directory
        if not os.path.exists(detector_path):
            detector_path = str(BASE_DIR.parent / 'best_20.pt')
        if not os.path.exists(descriptor_path):
            descriptor_path = str(BASE_DIR.parent / 'best_osnet_model.pth')
            
        detector = VehicleDetector(model_path=detector_path)
        descriptor = VehicleDescriptor(model_type='osnet', model_path=descriptor_path)
        
        # Use session-specific database path
        db_path = f"./chroma_vehicle_reid_streaming/{session_id}"
        classifier = ChromaDBVehicleReID(db_path=db_path)
        
        # Reset database for new session
        print(f"[Direct Processing] Resetting database for session {session_id}")
        classifier.reset_database()
        
        # Create pipeline (same as run_reid)
        pipeline = Pipeline(detector=detector, descriptor=descriptor, classifier=classifier)
        
        # Open video captures
        video_path1 = session_data["cameras"]["cam1"]
        video_path2 = session_data["cameras"]["cam2"]
        
        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)
        
        if not cap1.isOpened() or not cap2.isOpened():
            print(f"[Direct Processing] Error: Could not open video files")
            return
        
        # Get video properties
        fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
        fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
        
        print(f"[Direct Processing] Video1 FPS: {fps1}, Video2 FPS: {fps2}")
        
        # Target processing rate (6 FPS for web display)
        target_fps = 30
        frame_interval = 1.0 / target_fps
        
        frame_count = 0
        last_process_time = time.time()
        
        while session_id in active_sessions:
            current_time = time.time()
            
            # Frame rate control
            if current_time - last_process_time < frame_interval:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue
            
            last_process_time = current_time
            
            # Read frames from both cameras (synchronized like run_reid)
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 and not ret2:
                print(f"[Direct Processing] End of videos reached")
                break
            
            # Process frame1 (cam1) if available
            if ret1 and frame1 is not None:
                try:
                    # Process frame with the same pipeline as run_reid
                    result1 = pipeline.process(frame1, 'cam1')
                    
                    if result1 is not None:
                        # Encode processed frame for WebSocket transmission
                        _, buffer = cv2.imencode('.jpg', result1, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send to WebSocket
                        websocket_message = {
                            "type": "frame",
                            "camera": "cam1",
                            "data": frame_b64,
                            "timestamp": datetime.now().isoformat(),
                            "frame_count": frame_count + 1
                        }
                        
                        websocket_queue.put((session_id, websocket_message))
                        
                        # Clean up
                        del buffer, frame_b64, result1
                    
                    del frame1
                    
                except Exception as e:
                    print(f"[Direct Processing] Error processing cam1 frame: {e}")
            
            # Process frame2 (cam2) if available
            if ret2 and frame2 is not None:
                try:
                    # Process frame with thesame pipeline as run_reid
                    result2 = pipeline.process(frame2, 'cam2')
                    
                    if result2 is not None:
                        # Encode processed frame for WebSocket transmission
                        _, buffer = cv2.imencode('.jpg', result2, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send to WebSocket
                        websocket_message = {
                            "type": "frame",
                            "camera": "cam2",
                            "data": frame_b64,
                            "timestamp": datetime.now().isoformat(),
                            "frame_count": frame_count + 1
                        }
                        
                        websocket_queue.put((session_id, websocket_message))
                        
                        # Clean up
                        del buffer, frame_b64, result2
                    
                    del frame2
                    
                except Exception as e:
                    print(f"[Direct Processing] Error processing cam2 frame: {e}")
            
            frame_count += 1
            
            # Garbage collection and progress logging (same as run_reid)
            if frame_count % 100 == 0:
                gc.collect()
                if frame_count % 500 == 0:
                    print(f"[Direct Processing] Processed {frame_count} frames")
                    current_stats = classifier.get_statistics()
                    print(f"[Direct Processing] Current database size: {current_stats['total']} embeddings")
        
        print(f"[Direct Processing] Processing completed for session {session_id}")
        
    except Exception as e:
        print(f"[Direct Processing] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        try:
            cap1.release()
            cap2.release()
            if 'pipeline' in locals():
                pipeline.close()
            print(f"[Direct Processing] Resources cleaned up for session {session_id}")
        except Exception as e:
            print(f"[Direct Processing] Error during cleanup: {e}")

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
        matches = []
        
        # Debug log for troubleshooting
        # print(f"[Latest Matches] Looking for images in {matching_dir}")
        
        # Check for vehicle subdirectories in matching folder
        vehicle_dirs = [d for d in matching_dir.iterdir() if d.is_dir()]
        # print(f"[Latest Matches] Found {len(vehicle_dirs)} vehicle directories")
        
        # Process each vehicle directory
        for vehicle_dir in vehicle_dirs:
            # Get all image files in this vehicle directory
            vehicle_images = list(vehicle_dir.glob("*.jpg"))
            # print(f"[Latest Matches] Found {len(vehicle_images)} images in {vehicle_dir.name}")
            
            if len(vehicle_images) >= 2:  # Only include vehicles with matches from both cameras
                # Sort images by camera (cam1 first, then cam2)
                cam1_images = [img for img in vehicle_images if "cam1" in img.name.lower()]
                cam2_images = [img for img in vehicle_images if "cam2" in img.name.lower()]
                
                # Create match entries for each camera pair
                for cam1_img in cam1_images:
                    for cam2_img in cam2_images:
                        # Calculate relative paths from matching directory
                        relative_path1 = cam1_img.relative_to(matching_dir)
                        relative_path2 = cam2_img.relative_to(matching_dir)
                        
                        # Get the latest modification time from both images
                        latest_time = max(cam1_img.stat().st_mtime, cam2_img.stat().st_mtime)
                        
                        # Create match entry with both camera images
                        match_entry = {
                            "vehicle_id": vehicle_dir.name,
                            "cam1": {
                                "filename": cam1_img.name,
                                "track_id": cam1_img.stem.split('_')[-1][-1],  # Extract track ID from filename
                                "path": f"/matching/{relative_path1.as_posix()}",
                                "thumbnail_url": f"/matching_resized/{vehicle_dir.name}/{cam1_img.name}?width=90&height=120",
                                "small_thumbnail_url": f"/matching_resized/{vehicle_dir.name}/{cam1_img.name}?width=60&height=80"
                            },
                            "cam2": {
                                "filename": cam2_img.name,
                                "track_id": cam2_img.stem.split('_')[-1][-1],  # Extract track ID from filename
                                "path": f"/matching/{relative_path2.as_posix()}",
                                "thumbnail_url": f"/matching_resized/{vehicle_dir.name}/{cam2_img.name}?width=90&height=120",
                                "small_thumbnail_url": f"/matching_resized/{vehicle_dir.name}/{cam2_img.name}?width=60&height=80"
                            },
                            "timestamp": datetime.fromtimestamp(latest_time).isoformat(),
                            "match_time": latest_time
                        }
                        
                        matches.append(match_entry)
        
        # Sort by timestamp (newest first) and limit to 20 most recent matches
        matches.sort(key=lambda x: x["match_time"], reverse=True)
        matches = matches[:20]
        
        # Remove the internal match_time field before returning
        for match in matches:
            del match["match_time"]
        
        # print(f"[Latest Matches] Returning {len(matches)} vehicle matches")
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
    
    # Create session data (updated for direct processing)
    session_data = {
        "id": session_id,
        "status": "initialized",
        "created_at": datetime.now().isoformat(),
        "cameras": {"cam1": None, "cam2": None},
        "processing_thread": None,  # Replaces spark_thread and kafka_threads
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
    """Start processing the uploaded videos using direct pipeline (like run_reid)"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = active_sessions[session_id]
    
    # Check if both videos are uploaded
    if not session_data["cameras"]["cam1"] or not session_data["cameras"]["cam2"]:
        raise HTTPException(status_code=400, detail="Both camera videos must be uploaded")
    
    try:
        print(f"[Processing] Starting direct video processing for session {session_id}")
        
        # Start direct video processing thread (replaces Spark+Kafka)
        processing_thread = threading.Thread(target=direct_video_processing_thread, args=(session_id,))
        processing_thread.start()
        session_data["processing_thread"] = processing_thread
        
        # Start matching monitor thread
        matching_thread = threading.Thread(target=matching_monitor_thread, args=(session_id,))
        matching_thread.start()
        session_data["matching_thread"] = matching_thread
        
        # Update session status
        session_data["status"] = "processing"
        session_data["processing_start_time"] = datetime.now().isoformat()
        
        print(f"[Processing] Started direct processing for session {session_id}")
        
        return {
            "message": "Processing started with direct pipeline", 
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
    
    # Get thread status
    processing_thread_alive = False
    matching_thread_alive = False
    
    if session_data.get("processing_thread"):
        processing_thread_alive = session_data["processing_thread"].is_alive()
        
    if session_data.get("matching_thread"):
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
            "processing_alive": processing_thread_alive,
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
        "processing_mode": "direct_pipeline",
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
            "processing_thread": session_data["processing_thread"] is not None,
            "processing_thread_alive": session_data["processing_thread"].is_alive() if session_data["processing_thread"] else False,
            "matching_thread": session_data["matching_thread"] is not None,
            "matching_thread_alive": session_data["matching_thread"].is_alive() if session_data["matching_thread"] else False
        }
    
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
@app.get("/matching_resized/{vehicle_id}/{filename}")
async def get_resized_matching_image(vehicle_id: str, filename: str, width: int = 90, height: int = 120):
    """Serve a resized version of a matching image from a specific vehicle directory"""
    try:
        matching_dir = BASE_DIR.parent / "matching"
        
        print(f"[Resize] Looking for image: {filename} in vehicle directory: {vehicle_id}")
        
        # Construct the full path to the image
        vehicle_dir = matching_dir / vehicle_id
        image_path = vehicle_dir / filename
        
        # Check if the vehicle directory exists
        if not vehicle_dir.exists():
            print(f"[Resize] Vehicle directory not found: {vehicle_dir}")
            raise HTTPException(status_code=404, detail=f"Vehicle directory '{vehicle_id}' not found")
        
        # Check if the image file exists
        if not image_path.exists():
            print(f"[Resize] Image file not found: {image_path}")
            raise HTTPException(status_code=404, detail=f"Image '{filename}' not found in vehicle '{vehicle_id}'")
        
        print(f"[Resize] Found image: {image_path}")
        
        # Read and resize the image
        import cv2
        import numpy as np
        
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
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        print(f"Error serving resized image {vehicle_id}/{filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not process image")

# Fallback endpoint for backward compatibility (old URL structure)
@app.get("/matching_resized/{filename}")
async def get_resized_matching_image_fallback(filename: str, width: int = 90, height: int = 120):
    """Fallback endpoint for backward compatibility - searches all vehicle directories"""
    try:
        matching_dir = BASE_DIR.parent / "matching"
        
        # Clean filename by removing any camera prefix (cam1_, cam2_)
        clean_filename = filename
        if clean_filename.startswith(("cam1_", "cam2_")):
            clean_filename = clean_filename[5:]  # Remove the prefix
        
        print(f"[Resize Fallback] Looking for image: {clean_filename}, original request: {filename}")
        
        # Try to find the file in any vehicle subdirectory
        image_path = None
        vehicle_id = None
        
        # Search through all vehicle directories
        for subdir in matching_dir.iterdir():
            if subdir.is_dir():
                subdir_file = subdir / clean_filename
                if subdir_file.exists():
                    image_path = subdir_file
                    vehicle_id = subdir.name
                    print(f"[Resize Fallback] Found image in vehicle directory: {subdir_file}")
                    break
        
        if not image_path:
            print(f"[Resize Fallback] Image not found: {clean_filename}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Redirect to the new endpoint structure
        return await get_resized_matching_image(vehicle_id, clean_filename, width, height)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in fallback resize endpoint for {filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not process image")

@app.get("/test_resize")
async def test_resize():
    """Test endpoint to verify image resizing works with new folder structure"""
    try:
        matching_dir = BASE_DIR.parent / "matching"
        
        # Get the first available image from any vehicle directory
        test_image = None
        vehicle_id = None
        
        # Search through vehicle directories
        for subdir in matching_dir.iterdir():
            if subdir.is_dir():
                for img_path in subdir.glob("*.jpg"):
                    test_image = img_path
                    vehicle_id = subdir.name
                    break
            if test_image:
                break
        
        if test_image:
            # Calculate relative path from matching directory
            relative_path = test_image.relative_to(matching_dir)
            
            return {
                "success": True,
                "vehicle_id": vehicle_id,
                "test_image": test_image.name,
                "original_url": f"/matching/{relative_path.as_posix()}",
                "thumbnail_url": f"/matching_resized/{vehicle_id}/{test_image.name}?width=300&height=200",
                "small_thumbnail_url": f"/matching_resized/{vehicle_id}/{test_image.name}?width=150&height=100",
                "fallback_url": f"/matching_resized/{test_image.name}?width=300&height=200"
            }
        else:
            return {"success": False, "message": "No test images found"}
            
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

@app.get("/test_latest_matches")
async def test_latest_matches():
    """Test endpoint to verify latest matches works with new folder structure"""
    try:
        # Call the actual latest_matches endpoint
        result = await get_latest_matches()
        
        return {
            "success": True,
            "matches_found": len(result.get("matches", [])),
            "sample_match": result.get("matches", [])[0] if result.get("matches") else None,
            "structure": "vehicle_id -> cam1/cam2 images"
        }
        
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

@app.get("/inspect_matching_structure")
async def inspect_matching_structure():
    """Inspect the current matching folder structure for debugging"""
    try:
        matching_dir = BASE_DIR.parent / "matching"
        
        if not matching_dir.exists():
            return {"error": "Matching directory does not exist"}
        
        structure = {
            "matching_dir": str(matching_dir),
            "exists": True,
            "vehicle_directories": {}
        }
        
        # List all subdirectories (vehicle IDs)
        for subdir in matching_dir.iterdir():
            if subdir.is_dir():
                vehicle_images = []
                for img_file in subdir.glob("*.jpg"):
                    vehicle_images.append({
                        "filename": img_file.name,
                        "camera": "cam1" if "cam1" in img_file.name.lower() else "cam2" if "cam2" in img_file.name.lower() else "unknown",
                        "size": img_file.stat().st_size,
                        "modified": datetime.fromtimestamp(img_file.stat().st_mtime).isoformat()
                    })
                
                structure["vehicle_directories"][subdir.name] = {
                    "path": str(subdir),
                    "image_count": len(vehicle_images),
                    "images": vehicle_images
                }
        
        structure["total_vehicles"] = len(structure["vehicle_directories"])
        structure["total_images"] = sum(len(v["images"]) for v in structure["vehicle_directories"].values())
        
        return structure
        
    except Exception as e:
        return {"error": f"Failed to inspect structure: {str(e)}"}
    
@app.post("/test_direct_processing")
async def test_direct_processing():
    """Test endpoint to verify direct processing pipeline components"""
    try:
        # Test pipeline component imports
        from realtime_reid.vehicle_detector import VehicleDetector
        from realtime_reid.feature_extraction import VehicleDescriptor
        from realtime_reid.classifier_chromadb import ChromaDBVehicleReID
        from realtime_reid.pipeline import Pipeline
        
        test_results = {
            "imports_successful": True,
            "components_available": {
                "VehicleDetector": True,
                "VehicleDescriptor": True,
                "ChromaDBVehicleReID": True,
                "Pipeline": True
            },
            "model_files_check": {
                "vehicle_detector": os.path.exists("best_20.pt") or os.path.exists(str(BASE_DIR.parent / "best_20.pt")),
                "feature_extractor": os.path.exists("best_osnet_model.pth") or os.path.exists(str(BASE_DIR.parent / "best_osnet_model.pth"))
            },
            "processing_mode": "direct_pipeline",
            "message": "Direct processing pipeline is ready"
        }
        
        return test_results
        
    except Exception as e:
        return {
            "imports_successful": False,
            "error": str(e),
            "message": "Direct processing pipeline test failed"
        }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


