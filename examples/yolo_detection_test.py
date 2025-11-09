#!/usr/bin/env python3
"""
YOLO Detection Test with RealSense Camera

This script demonstrates real-time YOLO object detection using frames captured
from a RealSense camera. It uses the VisionWrapper class to perform two-stage
detection (ROI detection followed by object detection).

Features:
- Real-time RealSense camera capture
- Two-stage YOLO detection pipeline
- Live visualization with bounding boxes
- Frame rate monitoring
- Keyboard controls for capture and exit

Controls:
- 'c': Capture and save current frame
- 'q': Quit application
- 'r': Reset detection visualization
- 's': Save detection results to file

Author: Auto-generated test script
Date: November 8, 2025
"""

import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import json
from datetime import datetime

# Add parent directory to path to import wrapper modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from wrapper.wrapper_vision import VisionWrapper
except ImportError as e:
    print(f"Error importing VisionWrapper: {e}")
    print("Make sure you're running this from the project root or examples directory")
    sys.exit(1)


class YOLODetectionTest:
    def __init__(self, roi_model_path='../weights/obb/roi/best.pt', 
                 object_model_path='../weights/obb/object_detection/best.pt',
                 width=1280, height=720, fps=30):
        """
        Initialize the YOLO detection test with RealSense camera.
        
        Args:
            roi_model_path (str): Path to ROI detection model
            object_model_path (str): Path to object detection model
            width (int): Camera capture width
            height (int): Camera capture height
            fps (int): Camera capture FPS
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.detection_count = 0
        self.capture_count = 0
        
        # Initialize RealSense pipeline
        print("[INFO] Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        try:
            # Start streaming
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            
            # Get camera intrinsics and depth scale
            self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            color_stream = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            print(f"[INFO] Camera initialized successfully")
            print(f"[INFO] Depth scale: {self.depth_scale}")
            print(f"[INFO] Camera intrinsics: {self.intrinsics}")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize RealSense camera: {e}")
            sys.exit(1)
        
        # Initialize vision wrapper
        print("[INFO] Loading YOLO models...")
        try:
            self.vision_wrapper = VisionWrapper(
                roi_model_path=roi_model_path,
                object_model_path=object_model_path
            )
            print("[INFO] YOLO models loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO models: {e}")
            print("Please check that the model paths are correct:")
            print(f"  ROI model: {roi_model_path}")
            print(f"  Object model: {object_model_path}")
            sys.exit(1)
        
        # Create output directories
        self.output_dir = "detection_results"
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
        
    def capture_frame(self):
        """Capture a single frame from RealSense camera."""
        try:
            # Wait for a coherent pair of frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
        except Exception as e:
            print(f"[ERROR] Failed to capture frame: {e}")
            return None, None
    
    def run_detection(self, color_image, enable_vis=True):
        """Run YOLO detection on the captured frame."""
        try:
            # Run inference
            detections = self.vision_wrapper.run_inference(color_image, enable_vis=enable_vis)
            self.detection_count += len(detections)
            return detections
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return []
    
    def draw_info_overlay(self, image, detections, fps):
        """Draw information overlay on the image."""
        # Create info text
        info_text = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Detections: {len(detections)}",
            f"Total Detections: {self.detection_count}",
            f"Captures: {self.capture_count}",
            "",
            "Controls:",
            "  'c' - Capture frame",
            "  'r' - Reset counters", 
            "  's' - Save results",
            "  'q' - Quit"
        ]
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 220), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Draw text
        y_offset = 30
        for line in info_text:
            cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y_offset += 20
        
        return image
    
    def save_frame_and_results(self, color_image, depth_image, detections):
        """Save captured frame and detection results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Save color frame
        color_path = os.path.join(self.frames_dir, f"color_{timestamp}.png")
        cv2.imwrite(color_path, color_image)
        
        # Save depth frame (as grayscale visualization)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        depth_path = os.path.join(self.frames_dir, f"depth_{timestamp}.png")
        cv2.imwrite(depth_path, depth_colormap)
        
        # Save detection results
        results_data = {
            "timestamp": timestamp,
            "frame_count": self.frame_count,
            "color_image_path": color_path,
            "depth_image_path": depth_path,
            "detections": detections,
            "camera_intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.ppx,
                "cy": self.intrinsics.ppy,
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
                "depth_scale": self.depth_scale
            }
        }
        
        results_path = os.path.join(self.results_dir, f"results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.capture_count += 1
        print(f"[INFO] Saved frame and results: {timestamp}")
        return timestamp
    
    def calculate_fps(self):
        """Calculate and return current FPS."""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= 1.0:  # Update FPS every second
            self.avg_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        return self.avg_fps
    
    def reset_counters(self):
        """Reset all counters."""
        self.frame_count = 0
        self.detection_count = 0
        self.capture_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        print("[INFO] Counters reset")
    
    def save_session_summary(self):
        """Save a summary of the detection session."""
        summary = {
            "session_end": datetime.now().isoformat(),
            "total_frames": self.frame_count,
            "total_detections": self.detection_count,
            "captures_saved": self.capture_count,
            "average_fps": self.avg_fps,
            "camera_config": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "depth_scale": self.depth_scale
            }
        }
        
        summary_path = os.path.join(self.output_dir, f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[INFO] Session summary saved: {summary_path}")
    
    def run(self):
        """Main detection loop."""
        print("\n" + "="*50)
        print("YOLO Detection Test - RealSense Camera")
        print("="*50)
        print("Starting detection loop...")
        print("Press 'q' to quit, 'c' to capture, 'r' to reset, 's' to save results")
        print("="*50 + "\n")
        
        try:
            while True:
                # Capture frame
                color_image, depth_image = self.capture_frame()
                if color_image is None:
                    continue
                
                self.frame_count += 1
                
                # Run detection
                detections = self.run_detection(color_image, enable_vis=False)
                
                # Calculate FPS
                current_fps = self.calculate_fps()
                
                # Create display image (copy to avoid modifying original)
                display_image = color_image.copy()
                
                # Draw detection results manually for better control
                for det in detections:
                    # Draw bounding box
                    bbox = np.array(det['bbox']).astype(int)
                    cv2.polylines(display_image, [bbox], True, (0, 255, 0), 2)
                    
                    # Draw center point
                    center = det['center']
                    cv2.circle(display_image, center, 5, (0, 255, 0), -1)
                    
                    # Draw class label
                    cv2.putText(display_image, det['class'], 
                               (center[0] - 10, center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw info overlay
                display_image = self.draw_info_overlay(display_image, detections, current_fps)
                
                # Display the frame
                cv2.imshow('YOLO Detection Test - RealSense', display_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Quitting...")
                    break
                elif key == ord('c'):
                    self.save_frame_and_results(color_image, depth_image, detections)
                elif key == ord('r'):
                    self.reset_counters()
                elif key == ord('s'):
                    self.save_session_summary()
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("[INFO] Cleaning up...")
        try:
            self.pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()
        self.save_session_summary()
        print("[INFO] Cleanup complete")


def main():
    """Main function to run the detection test."""
    print("YOLO Detection Test with RealSense Camera")
    print("=========================================")
    
    # Check if model files exist
    roi_model = '/home/msis/Desktop/Label-Printer/weights/obb/roi/best.pt'
    obj_model = '/home/msis/Desktop/Label-Printer/weights/obb/object_detection/best.pt'
    
    if not os.path.exists(roi_model):
        print(f"[WARNING] ROI model not found: {roi_model}")
        print("Please ensure the model files are in the correct location")
    
    if not os.path.exists(obj_model):
        print(f"[WARNING] Object detection model not found: {obj_model}")
        print("Please ensure the model files are in the correct location")
    
    # Initialize and run detection test
    try:
        detector = YOLODetectionTest(
            roi_model_path=roi_model,
            object_model_path=obj_model,
            width=1280,
            height=720,
            fps=30
        )
        detector.run()
    except Exception as e:
        print(f"[ERROR] Failed to run detection test: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
