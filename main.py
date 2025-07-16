import cv2
import os
import gc
from modules.vehicle_detection import VehicleDetector
from modules.feature_extraction import VehicleDescriptor
from modules.reid_chromadb import ChromaDBVehicleReID
from modules.pipeline import Pipeline

def run_reid(video_path1, video_path2, 
                                   save_dir=None, 
                                   db_path="./chroma_vehicle_reid",
                                   process_every_n_frames=5,
                                   reset_database=False):
    # Initialize components with ChromaDB
    detector = VehicleDetector(model_path='best_20.pt')
    descriptor = VehicleDescriptor(model_type='osnet', model_path='best_osnet_model.pth')
    classifier = ChromaDBVehicleReID(db_path=db_path)
    
    # Reset database if requested
    if reset_database:
        print("Resetting database...")
        classifier.reset_database()
    
    # Show initial database statistics
    print("Initial database statistics:")
    stats = classifier.get_statistics()
    for vehicle_type, count in stats.items():
        if vehicle_type not in ['total', 'max_ids']:
            print(f"{vehicle_type}: {count} embeddings")
    print(f"Total: {stats['total']} embeddings")
    
    pipeline = Pipeline(detector=detector, descriptor=descriptor, classifier=classifier)
    
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Get video properties
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize frames
    max_width = 1920
    max_height = 1080
    
    if width1 > max_width or height1 > max_height:
        scale1 = min(max_width / width1, max_height / height1)
        width1 = int(width1 * scale1)
        height1 = int(height1 * scale1)
        resize1 = True
    else:
        resize1 = False
        
    if width2 > max_width or height2 > max_height:
        scale2 = min(max_width / width2, max_height / height2)
        width2 = int(width2 * scale2)
        height2 = int(height2 * scale2)
        resize2 = True
    else:
        resize2 = False
    
    # Create video writers
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        output_path1 = os.path.join(save_dir, 'cam1_reid_results.mp4')
        output_path2 = os.path.join(save_dir, 'cam2_reid_results.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out1 = cv2.VideoWriter(output_path1, fourcc, fps1//process_every_n_frames, (width1, height1))
        out2 = cv2.VideoWriter(output_path2, fourcc, fps2//process_every_n_frames, (width2, height2))

    frame_count = 0
    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 and not ret2:
                break
                
            if frame_count % process_every_n_frames == 0:
                if ret1:
                    # Resize frame if needed
                    if resize1:
                        frame1 = cv2.resize(frame1, (width1, height1))
                    
                    # Process frame
                    result1 = pipeline.process(frame1, 'cam1')
                    if save_dir and result1 is not None:
                        out1.write(result1)
                    
                    # Clean up
                    del frame1, result1
                        
                if ret2:
                    # Resize frame if needed
                    if resize2:
                        frame2 = cv2.resize(frame2, (width2, height2))
                    
                    # Process frame
                    result2 = pipeline.process(frame2, 'cam2')
                    if save_dir and result2 is not None:
                        out2.write(result2)
                    
                    # Clean up
                    del frame2, result2
                
                # Force garbage collection every 100 frames
                if frame_count % 100 == 0:
                    gc.collect()
                    if frame_count % 500 == 0:
                        print(f"Processed {frame_count} frames")
                        current_stats = classifier.get_statistics()
                        print(f"Current database size: {current_stats['total']} embeddings")

            frame_count += 1

    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Release video resources
        cap1.release()
        cap2.release()
        if save_dir:
            out1.release()
            out2.release()
        
        # Show final statistics
        print(f"\nProcessing completed. Total frames processed: {frame_count}")
        final_stats = classifier.get_statistics()
        print("Final database statistics:")
        for vehicle_type, count in final_stats.items():
            if vehicle_type not in ['total', 'max_ids']:
                print(f"  {vehicle_type}: {count} embeddings")
        print(f"Total: {final_stats['total']} embeddings")
        print(f"Max IDs: {final_stats['max_ids']}")
        
        # Close pipeline
        pipeline.close()

if __name__ == "__main__":
    video1 = r"D:\Khoa\University\3rdYear\BigData\Cam_1.mp4"
    video2 = r"D:\Khoa\University\3rdYear\BigData\Cam_2.mp4"
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run with memory optimization settings
    run_reid(
        video1, 
        video2, 
        save_dir=output_dir,
        reset_database=True,
    )
