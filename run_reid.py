import cv2
import os
from realtime_reid.vehicle_detector import VehicleDetector
from realtime_reid.feature_extraction import VehicleDescriptor
from realtime_reid.pipeline import Pipeline

def run_reid_on_two_videos(video_path1, video_path2, save_dir=None):
    detector = VehicleDetector(model_path='checkpoints/best_20.pt')
    descriptor = VehicleDescriptor(model_type='osnet', model_path='checkpoints/best_osnet_model.pth')
    pipeline = Pipeline(detector=detector, descriptor=descriptor)
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Get video properties
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writers
    if save_dir:
        output_path1 = os.path.join(save_dir, 'cam1_reid_results.mp4')
        output_path2 = os.path.join(save_dir, 'cam2_reid_results.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out1 = cv2.VideoWriter(output_path1, fourcc, fps1//5, (width1, height1))
        out2 = cv2.VideoWriter(output_path2, fourcc, fps2//5, (width2, height2))

    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 and not ret2:
            break
            
        # Process every 5th frame
        if frame_count % 5 == 0:
            if ret1:
                _, buffer1 = cv2.imencode('.jpg', frame1)
                img_bytes1 = buffer1.tobytes()
                result1 = pipeline.process(img_bytes1, 'cam1', save_dir=None, return_bytes=False)
                if save_dir:
                    out1.write(result1)
                    
            if ret2:
                _, buffer2 = cv2.imencode('.jpg', frame2)
                img_bytes2 = buffer2.tobytes()
                result2 = pipeline.process(img_bytes2, 'cam2', save_dir=None, return_bytes=False)
                if save_dir:
                    out2.write(result2)

        frame_count += 1

    # Release everything
    cap1.release()
    cap2.release()
    if save_dir:
        out1.release()
        out2.release()

if __name__ == "__main__":
    # Set your video paths and output directory here
    video1 = "data/cam1_v1.mp4"
    video2 = "data/cam2_v1.mp4"
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    run_reid_on_two_videos(video1, video2, save_dir=output_dir)
