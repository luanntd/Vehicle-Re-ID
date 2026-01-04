import sys
import argparse
import time
import os
import threading
from subprocess import Popen
from pathlib import Path

try:
    from subprocess import CREATE_NEW_CONSOLE
except Exception:
    CREATE_NEW_CONSOLE = 0

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    part for part in [os.environ.get("PYTHONPATH"), str(PROJECT_ROOT)] if part
)

from streaming.spark_streaming import start_spark_streaming
from modules.reid_chromadb import ChromaDBVehicleReID_spark   
crossed_vehicles = set()
def parse_args():
    parser = argparse.ArgumentParser(description='Start Spark Streaming job for Vehicle Re-ID')
    parser.add_argument("-t1", "--topic1",
                        type=str,
                        default="cam1",
                        help="Kafka topic name for camera 1")
    parser.add_argument("-t2", "--topic2",
                        type=str,
                        default="cam2",
                        help="Kafka topic name for camera 2")
    
    parser.add_argument("-c1", "--camera1",
                        type=str,
                        default=r"D:\Khoa\University\3rdYear\BigData\Cam_1.mp4",
                        help="Path to video file or image directory for camera 1")  
    parser.add_argument("-c2", "--camera2",
                        type=str,
                        default=r"D:\Khoa\University\3rdYear\BigData\Cam_2.mp4",
                        help="Path to video file or image directory for camera 2")
    
    parser.add_argument("-i", "--interval",
                        type=float,
                        default=0.2,
                        help="Frame interval in seconds for both cameras")
    
    parser.add_argument("-d", "--db_path",
                        type=str,
                        default="Vehicle-Re-ID/chroma_vehicle_reid_streaming",
                        help="Path to ChromaDB database")
    parser.add_argument("-cn", "--collection_name",
                        type=str,
                        default="vehicle_embeddings_streaming",
                        help="Name of the ChromaDB collection")
    
    parser.add_argument("--export_dir",
                        type=str,
                        default="Vehicle-Re-ID/matching_exports",
                        help="Directory to save cross-camera exports")
    parser.add_argument("--export_interval",
                        type=float,
                        default=3.0,
                        help="Seconds between cross-camera export sweeps (0 disables)")

    parser.add_argument("--view",
                        action="store_true", default=False,
                        help="Launch 2 Consumer.py subprocesses to view cam1_processed/cam2_processed")
    
    return parser.parse_args()


def start_viewers(args):
    """Start 2 viewer subprocesses (one per processed topic)."""
    cam1_processed = f"{args.topic1}_processed"
    cam2_processed = f"{args.topic2}_processed"
    print(f"[Main] Starting viewers for topics: {cam1_processed}, {cam2_processed}")

    consumer_script = str(PROJECT_ROOT / "Consumer.py")
    root_cwd = str(PROJECT_ROOT.parent)

    proc_cam1 = Popen([
        sys.executable,
        consumer_script,
        "-t",
        cam1_processed,
        "--view",
    ], cwd=root_cwd, creationflags=CREATE_NEW_CONSOLE)

    proc_cam2 = Popen([
        sys.executable,
        consumer_script,
        "-t",
        cam2_processed,
        "--view",
    ], cwd=root_cwd, creationflags=CREATE_NEW_CONSOLE)

    time.sleep(1)
    for name, proc in (("viewer-cam1", proc_cam1), ("viewer-cam2", proc_cam2)):
        code = proc.poll()
        if code is not None:
            print(f"[Main] {name} exited early with code {code}. Check its console output.")

    return proc_cam1, proc_cam2

def start_producers(args):
    print("[Main] Starting video producers...")
    producer1_cmd = f'{sys.executable} Vehicle-Re-ID\\Producer.py -t {args.topic1} -c {args.camera1} -i {args.interval}'
    producer2_cmd = f'{sys.executable} Vehicle-Re-ID\\Producer.py -t {args.topic2} -c {args.camera2} -i {args.interval}'
    
    proc1 = Popen(producer1_cmd, shell=False)
    proc2 = Popen(producer2_cmd, shell=False)
    
    return proc1, proc2

def start_chromadb_server(db_path):
    print("[Main] Starting ChromaDB server...")
    chroma_cmd = f'chroma run --path {db_path}' if db_path else 'chroma start'
    chroma_proc = Popen(chroma_cmd, shell=False)
    return chroma_proc

def _export_cross_camera_matches(chroma_client, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    max_ids = chroma_client._load_max_ids()
    print(f"[Main] Cross-camera export max IDs: {max_ids}")
    for vehicle_type_id, vehicle_type_name in chroma_client.VEHICLE_TYPES.items():
        upper_bound = max_ids.get(vehicle_type_id, 0)
        for vehicle_id in range(upper_bound):
            try:
                matches = chroma_client.get_cross_camera_matches(vehicle_id, vehicle_type_id)
                if not matches or len(matches) < 2:
                    continue
                if (vehicle_type_id, vehicle_id) in crossed_vehicles:
                    continue
                print(f"[Main] Found cross-camera matches for {vehicle_type_name} #{vehicle_id}")
                crossed_vehicles.add((vehicle_type_id, vehicle_id))
                chroma_client.save_cross_camera_images(
                    vehicle_id=vehicle_id,
                    vehicle_type=vehicle_type_id,
                    camera_matches=matches,
                    save_dir=export_dir,
                )
                print(f"[Main] Saved cross-camera matches for {vehicle_type_name} #{vehicle_id}")
            except Exception as export_error:
                print(f"[Main] Cross-camera export error for {vehicle_type_name} #{vehicle_id}: {export_error}")


def start_cross_camera_export_scheduler(args):
    if args.export_interval <= 0:
        return None, None

    stop_event = threading.Event()

    def _run_loop():
        print(f"[Main] Cross-camera export loop running every {args.export_interval}s")
        chroma_client = ChromaDBVehicleReID_spark(
            db_path=args.db_path or "",
            collection_name=args.collection_name,
        )
        chroma_client.reset_database()
        time.sleep(10) 
        while not stop_event.is_set():
            print("[Main] Running cross-camera export sweep...")
            _export_cross_camera_matches(chroma_client, args.export_dir)
            stop_event.wait(args.export_interval)
        print("[Main] Cross-camera export loop stopped")

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    return stop_event, thread

def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    metrics = {
        "total_rows": 0,
        "total_batch_time": 0.0,
        "total_runtime_s": 0.0
    }

    per_camera_latest = {}
    def _progress_callback(payload):
        if payload.get("type") == "per_camera_metrics":
            per_camera_latest.update(payload.get("per_camera") or {})
            metrics["total_runtime_s"] += payload.get("batch_duration_s")
            return
        rows = payload.get("num_input_rows", 0) or 0
        duration_ms = payload.get("duration_ms")
        
        if not rows or duration_ms is None:
            return
        metrics["total_rows"] += rows
        metrics["total_batch_time"] += duration_ms / 1000.0
    
    # Start ChromaDB server
    chroma_proc = start_chromadb_server(args.db_path)
    chroma_client = ChromaDBVehicleReID_spark(
        db_path=args.db_path or "",
        collection_name=args.collection_name,
    )
    print("[Main] Resetting ChromaDB database for streaming...")
    chroma_client.reset_database()
    proc1 = proc2 = None
    view1 = view2 = None
    spark_thread = None


    try:
        proc1, proc2 = start_producers(args)

        print("[Main] Starting Spark Streaming job...")
        spark_thread = threading.Thread(
            target=start_spark_streaming,
            kwargs={
                "base_dir": base_dir,
                "progress_callback": _progress_callback,
            },
            daemon=True,
        )
        spark_thread.start()

        if getattr(args, "view", False):
            view1, view2 = start_viewers(args)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("[Main] Interrupted. Shutting down...")
        
    finally:
        if per_camera_latest:
            for cam, stats in per_camera_latest.items():
                total_processed = stats.get("total_processed") or 0
                mean_e2e_ms = stats.get("total_mean_e2e_latency_ms")
                mean_e2e_fps = (total_processed * 1000 / mean_e2e_ms) if total_processed else 0.0

                first_capture_ts_ms = stats.get("first_capture_ts_ms")
                last_capture_ts_ms = stats.get("last_capture_ts_ms")
                last_spark_output_ts_ms = stats.get("last_spark_output_ts_ms")

                send_span_s = None
                spark_finish_span_s = None
                total_span_s = None

                if first_capture_ts_ms is not None and last_capture_ts_ms is not None:
                    send_span_s = (float(last_capture_ts_ms) - float(first_capture_ts_ms)) / 1000.0
                if last_capture_ts_ms is not None and last_spark_output_ts_ms is not None:
                    spark_finish_span_s = (float(last_spark_output_ts_ms) - float(last_capture_ts_ms)) / 1000.0
                if first_capture_ts_ms is not None and last_spark_output_ts_ms is not None:
                    total_span_s = (float(last_spark_output_ts_ms) - float(first_capture_ts_ms)) / 1000.0

                print(
                    f"[Main] {cam}: mean_fps={mean_e2e_fps:.2f}, "
                    f"mean_latency_ms={mean_e2e_ms if mean_e2e_ms is not None else 'n/a'}"
                    f" over {total_processed} frames; "
                    f"send_span_s={send_span_s if send_span_s is not None else 'n/a'}, "
                    f"spark_finish_span_s={spark_finish_span_s if spark_finish_span_s is not None else 'n/a'}, "
                    f"total_span_s={total_span_s if total_span_s is not None else 'n/a'},"
                )
        else:
            print("[Main] No per-camera e2e metrics collected.")
        print("Total runtime seconds: ", metrics["total_runtime_s"])

        total_rows = metrics["total_rows"]
        total_time = metrics["total_batch_time"]
        if total_rows and total_time:
            mean_fps = total_rows / total_time
            mean_latency_ms = (total_time / total_rows) * 1000.0
            print(f"[Main] Mean FPS: {mean_fps:.2f}, Mean latency: {mean_latency_ms:.2f} ms/frame")
        else:
            print("[Main] No streaming metrics collected.")
        # Terminate all processes
        for proc in (view1, view2, proc1, proc2):
            if proc:
                proc.terminate()
        print("[Main] All processes terminated.")
if __name__ == '__main__':
    main()