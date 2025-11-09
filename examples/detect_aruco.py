#!/usr/bin/env python3
"""
ArUco Marker Detection in Real-World Coordinates

This module provides ArUco marker detection functionality that transforms
detected marker positions to real-world coordinates using pre-calibrated
hand-eye transformation matrices and camera intrinsics.

Features:
- Real-time ArUco marker detection from RealSense camera
- Transform pixel coordinates to real-world robot base coordinates
- Support for multiple ArUco dictionaries
- Robust depth estimation using median filtering
- Visualization with marker annotations
- Integration with existing hand-eye calibration data

Author: Auto-generated from calibration reference
Date: November 9, 2025
"""

import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import time
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wrapper.wrapper_robot import SimpleRobot 
from coordinate_transform.handeye import HandeyeTransformer, get_robot_transform_matrix


@dataclass
class ArucoDetection:
    """Data class for storing ArUco detection results"""
    marker_id: int
    center_pixel: Tuple[float, float]
    corners_pixel: np.ndarray
    center_world: Optional[Tuple[float, float, float]] = None
    depth_m: Optional[float] = None
    confidence: Optional[float] = None


class ArucoDetector:
    """
    ArUco marker detector with real-world coordinate transformation capability.
    
    This class integrates ArUco marker detection with hand-eye calibration to provide
    real-world coordinates of detected markers relative to the robot base frame.
    """
    
    def __init__(self, 
                 aruco_dict_type=cv2.aruco.DICT_4X4_100,
                 camera_width=1280, 
                 camera_height=720,
                 camera_fps=30):
        """
        Initialize the ArUco detector.
        
        Args:
            aruco_dict_type: OpenCV ArUco dictionary type
            camera_width: Camera resolution width
            camera_height: Camera resolution height  
            camera_fps: Camera frame rate
        """
        # self.robot_real_instance = SimpleRobot(ip = "192.168.2.36")

        self.aruco_dict_type = aruco_dict_type
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # Initialize camera
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.camera_intrinsics = None
        
        # Initialize hand-eye transformer
        self.handeye_transformer = HandeyeTransformer()
        self._load_calibration_data()
        
        # Detection history for filtering
        self.detection_history = []
        self.max_history = 10
        
    def _load_calibration_data(self):
        """Load hand-eye calibration data and camera intrinsics"""
        try:
            # Load transformation matrices
            T_cam2gripper_file = "/home/msis/Desktop/Label-Printer/handeye_calibration_data/FinalTransforms/T_cam2gripper_HORAUD.npz"
            intrinsic_matrix_file = "/home/msis/Desktop/Label-Printer/handeye_calibration_data/FinalTransforms/IntrinsicMatrix.npz"
            
            if not os.path.exists(T_cam2gripper_file):
                # Try alternative path
                T_cam2gripper_file = "/home/msis/Desktop/Label-Printer/calibration/matrixs/T_cam2gripper_HORAUD.npz"
                intrinsic_matrix_file = "/home/msis/Desktop/Label-Printer/calibration/matrixs/IntrinsicMatrix.npz"
            
            self.handeye_transformer.load_calibration(T_cam2gripper_file)
            self.handeye_transformer.load_intrinsics(intrinsic_matrix_file)
            
            print(f"[INFO] Successfully loaded calibration data")
            print(f"[INFO] T_cam2gripper: {T_cam2gripper_file}")
            print(f"[INFO] Intrinsic matrix: {intrinsic_matrix_file}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load calibration data: {e}")
            raise
    
    def initialize_camera(self):
        """Initialize RealSense camera pipeline"""
        try:
            print("[INFO] Initializing RealSense camera...")
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure streams
            config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, self.camera_fps)
            config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, self.camera_fps)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            
            # Get camera parameters
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            color_stream = profile.get_stream(rs.stream.color)
            self.camera_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            print(f"[INFO] Camera initialized successfully")
            print(f"[INFO] Resolution: {self.camera_width}x{self.camera_height}")
            print(f"[INFO] Depth scale: {self.depth_scale}")
            print(f"[INFO] Camera intrinsics: fx={self.camera_intrinsics.fx:.1f}, fy={self.camera_intrinsics.fy:.1f}")
            
            # Warm up camera
            for _ in range(10):
                self.pipeline.wait_for_frames()
            print("[INFO] Camera warmup complete")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize camera: {e}")
            raise
    
    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[rs.depth_frame]]:
        """
        Capture aligned color and depth frames from camera.
        
        Returns:
            Tuple of (color_image, depth_frame) or (None, None) if capture fails
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            return color_image, depth_frame
            
        except Exception as e:
            print(f"[ERROR] Frame capture failed: {e}")
            return None, None
    
    def detect_markers(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect ArUco markers in the given image.
        
        Args:
            image: Input color image (BGR format)
            
        Returns:
            Tuple of (corners, ids) from ArUco detection
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        return corners, ids
    
    def get_robust_depth(self, depth_frame: rs.depth_frame, u: int, v: int, window_size: int = 5) -> Optional[float]:
        """
        Get robust depth estimate using median filtering in a local window.
        
        Args:
            depth_frame: RealSense depth frame
            u, v: Pixel coordinates
            window_size: Size of the sampling window
            
        Returns:
            Median depth in meters, or None if no valid depth found
        """
        half_window = window_size // 2
        depths = []
        
        # Sample depths in window around the point
        for dy in range(-half_window, half_window + 1):
            for dx in range(-half_window, half_window + 1):
                x = u + dx
                y = v + dy
                
                # Check bounds
                if 0 <= x < self.camera_width and 0 <= y < self.camera_height:
                    depth = depth_frame.get_distance(x, y)
                    if depth > 0:  # Valid depth
                        depths.append(depth)
        
        if not depths:
            return None
        
        return float(np.median(depths))
    
    def pixel_to_world_coordinates(self, u: float, v: float, depth_m: float, robot_tcp_pose) -> Optional[np.ndarray]:
        """
        Transform pixel coordinates to real-world robot base coordinates.
        
        Args:
            u, v: Pixel coordinates
            depth_m: Depth in meters
            robot_tcp_pose: Current robot TCP pose
            
        Returns:
            3D coordinates in robot base frame [x, y, z] in meters, or None if transformation fails
        """
        try:
            # Create RealSense intrinsics object with calibrated parameters
            rs_intrinsics = rs.intrinsics()
            rs_intrinsics.width = self.camera_width
            rs_intrinsics.height = self.camera_height
            rs_intrinsics.fx = float(self.handeye_transformer.intrinsic_matrix[0, 0])
            rs_intrinsics.fy = float(self.handeye_transformer.intrinsic_matrix[1, 1])
            rs_intrinsics.ppx = float(self.handeye_transformer.intrinsic_matrix[0, 2])
            rs_intrinsics.ppy = float(self.handeye_transformer.intrinsic_matrix[1, 2])
            rs_intrinsics.model = rs.distortion.none
            
            # Add distortion coefficients if available
            if self.handeye_transformer.dist_coeffs is not None:
                for i in range(min(5, len(self.handeye_transformer.dist_coeffs.flatten()))):
                    rs_intrinsics.coeffs[i] = float(self.handeye_transformer.dist_coeffs.flatten()[i])
            
            # Convert pixel to camera coordinates
            P_cam = np.array(rs.rs2_deproject_pixel_to_point(rs_intrinsics, [float(u), float(v)], float(depth_m))).reshape(3, 1)
            P_cam_h = np.vstack((P_cam, [1.0]))
            
            # Get robot transformation matrix
            T_base2ee = get_robot_transform_matrix(robot_tcp_pose)
            
            # Transform to base frame: P_base = T_base2ee @ T_cam2gripper @ P_cam
            P_base_h = T_base2ee @ self.handeye_transformer.T_cam2gripper @ P_cam_h
            P_base = P_base_h[:3].flatten()
            
            return P_base
            
        except Exception as e:
            print(f"[ERROR] Coordinate transformation failed: {e}")
            return None
    
    def detect_and_transform(self, robot_tcp_pose, enable_visualization=True) -> List[ArucoDetection]:
        """
        Detect ArUco markers and transform to real-world coordinates.
        
        Args:
            robot_tcp_pose: Current robot TCP pose for coordinate transformation
            enable_visualization: Whether to show detection visualization
            
        Returns:
            List of ArucoDetection objects with pixel and world coordinates
        """
        # Capture frame
        color_image, depth_frame = self.capture_frame()
        if color_image is None or depth_frame is None:
            print("[WARNING] Failed to capture valid frames")
            return []
        
        # Detect markers
        corners, ids = self.detect_markers(color_image)
        
        detections = []
        if ids is not None and len(ids) > 0:
            # Process each detected marker
            ids = ids.flatten()
            
            for marker_id, marker_corners in zip(ids, corners):
                try:
                    # Calculate marker center
                    pts = marker_corners.reshape(-1, 2)
                    center_px = pts.mean(axis=0)
                    u, v = int(round(center_px[0])), int(round(center_px[1]))
                    
                    # Get robust depth estimate
                    depth_m = self.get_robust_depth(depth_frame, u, v, window_size=5)
                    
                    if depth_m is None or depth_m <= 0:
                        print(f"[WARNING] Invalid depth for marker {marker_id}")
                        continue
                    
                    # Transform to world coordinates
                    world_coords = self.pixel_to_world_coordinates(center_px[0], center_px[1], depth_m, robot_tcp_pose)
                    
                    if world_coords is not None:
                        # Create detection object
                        detection = ArucoDetection(
                            marker_id=int(marker_id),
                            center_pixel=(float(center_px[0]), float(center_px[1])),
                            corners_pixel=pts,
                            center_world=(float(world_coords[0]), float(world_coords[1]), float(world_coords[2])),
                            depth_m=depth_m
                        )
                        detections.append(detection)
                        
                        print(f"[SUCCESS] Marker {marker_id}:")
                        print(f"  Pixel: ({center_px[0]:.1f}, {center_px[1]:.1f})")
                        print(f"  Depth: {depth_m:.4f} m")
                        print(f"  World: ({world_coords[0]:.4f}, {world_coords[1]:.4f}, {world_coords[2]:.4f}) m")
                        print(f"  World (mm): ({world_coords[0]*1000:.1f}, {world_coords[1]*1000:.1f}, {world_coords[2]*1000:.1f}) mm")
                
                except Exception as e:
                    print(f"[ERROR] Processing marker {marker_id}: {e}")
                    continue
        
        # Visualization
        if enable_visualization and len(detections) > 0:
            self.visualize_detections(color_image, detections, corners, ids)
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[ArucoDetection], corners, ids):
        """
        Visualize detected markers with annotations.
        
        Args:
            image: Input color image
            detections: List of detection results
            corners: ArUco corners from detection
            ids: ArUco IDs from detection
        """
        # Create output image
        vis_image = image.copy()
        
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(vis_image, corners, ids.reshape(-1, 1))
        
        # Add custom annotations
        for detection in detections:
            u, v = int(detection.center_pixel[0]), int(detection.center_pixel[1])
            
            # Draw center point
            cv2.circle(vis_image, (u, v), 8, (0, 255, 0), -1)
            cv2.circle(vis_image, (u, v), 12, (255, 255, 255), 2)
            
            # Add text annotations
            if detection.center_world is not None:
                world_text = f"ID:{detection.marker_id} ({detection.center_world[0]*1000:.1f}, {detection.center_world[1]*1000:.1f}, {detection.center_world[2]*1000:.1f})mm"
                depth_text = f"Depth: {detection.depth_m:.3f}m"
                
                # Background rectangle for text
                text_size = cv2.getTextSize(world_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_image, (u - 10, v - 40), (u + text_size[0] + 10, v + 10), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(vis_image, world_text, (u - 5, v - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis_image, depth_text, (u - 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add info overlay
        info_text = [
            f"Detected: {len(detections)} markers",
            f"Dictionary: {self.aruco_dict_type}",
            "Press 'q' to quit, 'c' to capture"
        ]
        
        y_offset = 30
        for line in info_text:
            cv2.putText(vis_image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        # Display image
        cv2.imshow('ArUco Detection - Real World Coordinates', vis_image)
    
    def run_continuous_detection(self, robot_tcp_pose_callback, save_callback=None):
        """
        Run continuous ArUco detection with live visualization.
        
        Args:
            robot_tcp_pose_callback: Function that returns current robot TCP pose
            save_callback: Optional callback for saving detections
        """
        print("\n" + "="*60)
        print("ArUco Detection - Real World Coordinates")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Capture current detections") 
        print("  's' - Save detection data")
        print("  'r' - Reset detection history")
        print("="*60 + "\n")
        
        detection_count = 0
        frame_count = 0
        
        try:
            while True:
                frame_count += 1
                
                # Get current robot pose
                try:
                    robot_pose = robot_tcp_pose_callback()
                except Exception as e:
                    print(f"[ERROR] Failed to get robot pose: {e}")
                    robot_pose = [0, 0, 0, 0, 0, 0]  # Default pose
                
                # Detect markers
                detections = self.detect_and_transform(robot_pose, enable_visualization=True)
                
                if detections:
                    detection_count += len(detections)
                    
                    # Add to history
                    self.detection_history.append({
                        'frame': frame_count,
                        'timestamp': time.time(),
                        'robot_pose': robot_pose,
                        'detections': detections
                    })
                    
                    # Limit history size
                    if len(self.detection_history) > self.max_history:
                        self.detection_history.pop(0)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Quitting...")
                    break
                elif key == ord('c'):
                    print(f"\n[CAPTURE] Frame {frame_count} - {len(detections)} markers detected")
                    for det in detections:
                        if det.center_world:
                            print(f"  Marker {det.marker_id}: {det.center_world} m")
                elif key == ord('s') and save_callback:
                    save_callback(self.detection_history)
                elif key == ord('r'):
                    self.detection_history.clear()
                    detection_count = 0
                    frame_count = 0
                    print("[INFO] Detection history reset")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def save_detections(self, detections: List[ArucoDetection], filename: Optional[str] = None) -> str:
        """
        Save detection results to file.
        
        Args:
            detections: List of detection results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"aruco_detections_{timestamp}.json"
        
        import json
        
        # Convert to serializable format
        data = {
            'timestamp': time.time(),
            'detections': []
        }
        
        for det in detections:
            det_data = {
                'marker_id': det.marker_id,
                'center_pixel': det.center_pixel,
                'corners_pixel': det.corners_pixel.tolist(),
                'center_world': det.center_world,
                'depth_m': det.depth_m
            }
            data['detections'].append(det_data)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Saved {len(detections)} detections to {filename}")
        return filename
    
    def cleanup(self):
        """Clean up resources"""
        print("[INFO] Cleaning up...")
        try:
            if self.pipeline:
                self.pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")

robot = SimpleRobot(ip = "192.168.2.36")
def demo_robot_pose_callback():
    """Dummy robot pose callback for testing"""
    
    # Return a fixed pose for demo purposes
    # In real use, this should call actual robot controller
    return robot.getcurrent_TCP()  # [x, y, z, rx, ry, rz] in mm and degrees


def demo_save_callback(history):
    """Demo callback for saving detection history"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"detection_history_{timestamp}.json"
    
    import json
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"[INFO] Saved detection history to {filename}")


def main():
    """Main function for standalone testing"""
    print("ArUco Detection in Real-World Coordinates")
    print("="*50)
    
    try:
        # Initialize detector
        detector = ArucoDetector(
            aruco_dict_type=cv2.aruco.DICT_4X4_100,
            camera_width=1280,
            camera_height=720,
            camera_fps=30
        )
        
        # Initialize camera
        detector.initialize_camera()
        
        # Run continuous detection with demo callbacks
        detector.run_continuous_detection(
            robot_tcp_pose_callback=demo_robot_pose_callback,
            save_callback=demo_save_callback
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to run ArUco detection: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
