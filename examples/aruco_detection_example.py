#!/usr/bin/env python3
"""
ArUco Detection Integration Example

This example shows how to integrate ArUco detection with the existing
robot system for real-world coordinate detection.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.detect_aruco import ArucoDetector
from rb5.cobot import SimpleRobot


def robot_pose_callback():
    """Get current robot TCP pose from the robot controller"""
    try:
        # Initialize robot (you may want to keep this as a class member for efficiency)
        robot = SimpleRobot(ip="192.168.2.36")  # Use your robot IP
        tcp_info = robot.getcurrent_TCP()
        
        # Handle different return formats from robot controller
        if isinstance(tcp_info, tuple) and len(tcp_info) >= 2:
            return tcp_info[1]  # Extract pose array from (success, pose) tuple
        elif isinstance(tcp_info, list):
            return tcp_info
        else:
            print(f"[WARNING] Unexpected TCP info format: {type(tcp_info)}")
            return [0, 0, 500, 90, 0, 0]  # Default fallback pose
    
    except Exception as e:
        print(f"[ERROR] Failed to get robot pose: {e}")
        return [0, 0, 500, 90, 0, 0]  # Default fallback pose


def detection_save_callback(detection_history):
    """Save detection history to file"""
    import json
    import time
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"aruco_detection_session_{timestamp}.json"
    
    # Convert detection history to serializable format
    serializable_history = []
    for entry in detection_history:
        serializable_entry = {
            'frame': entry['frame'],
            'timestamp': entry['timestamp'],
            'robot_pose': entry['robot_pose'],
            'detections': []
        }
        
        for det in entry['detections']:
            det_data = {
                'marker_id': det.marker_id,
                'center_pixel': det.center_pixel,
                'corners_pixel': det.corners_pixel.tolist(),
                'center_world': det.center_world,
                'depth_m': det.depth_m
            }
            serializable_entry['detections'].append(det_data)
        
        serializable_history.append(serializable_entry)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    print(f"[INFO] Saved detection session to {filename}")


def single_detection_example():
    """Example of single-shot detection"""
    print("Single Detection Example")
    print("=" * 30)
    
    # Initialize detector
    detector = ArucoDetector(
        aruco_dict_type=cv2.aruco.DICT_4X4_100,
        camera_width=1280,
        camera_height=720
    )
    
    try:
        # Initialize camera
        detector.initialize_camera()
        
        # Get current robot pose
        robot_pose = robot_pose_callback()
        print(f"Robot pose: {robot_pose}")
        
        # Single detection
        detections = detector.detect_and_transform(robot_pose, enable_visualization=True)
        
        if detections:
            print(f"\nDetected {len(detections)} ArUco markers:")
            for det in detections:
                if det.center_world:
                    print(f"  Marker {det.marker_id}:")
                    print(f"    Pixel: ({det.center_pixel[0]:.1f}, {det.center_pixel[1]:.1f})")
                    print(f"    World (m): ({det.center_world[0]:.4f}, {det.center_world[1]:.4f}, {det.center_world[2]:.4f})")
                    print(f"    World (mm): ({det.center_world[0]*1000:.1f}, {det.center_world[1]*1000:.1f}, {det.center_world[2]*1000:.1f})")
            
            # Save results
            detector.save_detections(detections)
        else:
            print("No ArUco markers detected")
        
        # Keep visualization open until key press
        import cv2
        cv2.waitKey(0)
        
    finally:
        detector.cleanup()


def continuous_detection_example():
    """Example of continuous detection with robot integration"""
    print("Continuous Detection Example")
    print("=" * 35)
    
    # Initialize detector
    detector = ArucoDetector(
        aruco_dict_type=cv2.aruco.DICT_4X4_100,
        camera_width=1280,
        camera_height=720
    )
    
    try:
        # Initialize camera
        detector.initialize_camera()
        
        # Run continuous detection
        detector.run_continuous_detection(
            robot_tcp_pose_callback=robot_pose_callback,
            save_callback=detection_save_callback
        )
        
    except Exception as e:
        print(f"[ERROR] Continuous detection failed: {e}")
    finally:
        detector.cleanup()


def main():
    """Main function with menu selection"""
    print("ArUco Detection Integration Examples")
    print("=" * 40)
    print("1. Single detection example")
    print("2. Continuous detection with robot")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                single_detection_example()
                break
            elif choice == '2':
                continuous_detection_example()
                break
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import cv2  # Import here to avoid issues if opencv not available
    main()
