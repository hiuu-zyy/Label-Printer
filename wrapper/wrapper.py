# wrapper.py

import json
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wrapper_robot import SimpleRobot 
from wrapper_vision import VisionWrapper
from coordinate_transform.handeye import HandeyeTransformer
import yaml
from loguru import logger

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)

def euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg):
    """Euler intrinsic ZYX: R = Rz(rz) @ Ry(ry) @ Rx(rx)"""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    return Rz(rz) @ Ry(ry) @ Rx(rx)



class TaskPlanner:
    def __init__(self, roi_model_path, object_model_path, camera_intrinsics, config_path=None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        self.LP_tilt_angle = self.config.get('label_printer', {}).get('tilt_angle', 45)

        # self.Gripper_real = Gripper_real()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 5)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 5)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.camera_intrinsics = camera_intrinsics

        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Initialize real robot instance
        self.robot_real_instance = SimpleRobot(ip = "192.168.2.36")

        # Initialize vision wrapper
        self.vision_wrapper = VisionWrapper(roi_model_path=roi_model_path, object_model_path=object_model_path)
        
        # Initialize handeye transformer
        self.handeye_transformer = HandeyeTransformer()

    def screw_hole_detection(self, max_attempts=50, retry_delay=0.1):
        """
        Detect screw holes with retry logic until ROIs are found.
        
        Args:
            max_attempts (int): Maximum number of detection attempts
            retry_delay (float): Delay between attempts in seconds
        
        Returns:
            real_world_coordinates: List of 3D coordinates of detected holes
        """
        attempt = 0
        print(f"[INFO] Starting screw hole detection...")
        
        while attempt < max_attempts:
            attempt += 1
            print(f"[INFO] Detection attempt {attempt}/{max_attempts}")
            
            # Capture fresh frames from the camera
            for _ in range(5):
                frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print(f"[WARNING] Failed to get valid frames on attempt {attempt}")
                time.sleep(retry_delay)
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Run inference using VisionWrapper
            detection_results = self.vision_wrapper.run_inference(color_image, enable_vis=True)
            
            # Check if any ROIs were detected by looking for any detections
            if not detection_results:
                print(f"[WARNING] No ROIs detected on attempt {attempt}, retrying...")
                time.sleep(retry_delay)
                continue
            
            # Process detections to find holes
            points = []
            hole_count = 0
            for det in detection_results:
                if det['class'] == 'H':  # Changed from 'label' to 'class' based on VisionWrapper output
                    cx, cy = det['center']
                    depth = depth_image[cy, cx] * self.depth_scale
                    points.append((cx, cy, depth))
                    hole_count += 1
            
            if hole_count > 0:
                print(f"[SUCCESS] Found {hole_count} hole(s) on attempt {attempt}")
                real_world_coordinates = self.handeye_transformer.transform(
                    point=points, 
                    depth_image=depth_image, 
                    camera_intrinsics=None,  # Will use calibrated intrinsics
                    depth_scale=self.depth_scale, 
                    robot_current_tcp=self.robot_real_instance.getcurrent_TCP()
                )
                return real_world_coordinates
            else:
                print(f"[WARNING] ROIs detected but no holes found on attempt {attempt}, retrying...")
                time.sleep(retry_delay)
        
        # If we get here, we've exhausted all attempts
        print(f"[ERROR] Failed to detect any screw holes after {max_attempts} attempts")
        raise RuntimeError(f"No screw holes detected after {max_attempts} attempts. Please check camera positioning and lighting.")
    
    def ready_position(self, real_world_coordinates):
        pair_pos = []
        for p in real_world_coordinates:
            down = [p[0], p[1], p[2], self.robot_real_instance.getcurrent_TCP()[3], self.robot_real_instance.getcurrent_TCP()[4], self.robot_real_instance.getcurrent_TCP()[5]]
            logger.debug(f"Screw down position: {down}")
            # down = [-200, -800, 300, 90, 0, 0]
            T_ee2base = self.get_robot_transform_matrix(down)
            local_point_homogeneous = np.array([[0, 100, 0, 1]]).T
            base_point = T_ee2base @ local_point_homogeneous
            base_point = base_point.flatten()
            x_base, y_base, z_base = base_point[0], base_point[1], base_point[2]
            rx, ry, rz = self.robot_real_instance.getcurrent_TCP()[3:]
            x_base =round(x_base,2)
            y_base =round(y_base,2)
            z_base =round(z_base,2)
            logger.debug(f"Screw down position calculated: {[x_base, y_base, z_base, rx, ry, rz]}")
            up = ([x_base, y_base, z_base, rx, ry, rz])
            pair = [down, up]
            pair_pos.append(pair)

        return pair_pos

    def get_robot_transform_matrix(self, tcp_pose):
        """Convert robot TCP pose to 4x4 transformation matrix
        
        Args:
            tcp_pose: Can be:
                - tuple: (success, [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg])
                - dict: {'pos': [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]}
                - list/tuple: [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        
        Returns:
            T_base2ee: 4x4 transformation matrix
        """
        if isinstance(tcp_pose, tuple) and len(tcp_pose) >= 2:
            position_array = tcp_pose[1]
            x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = position_array
        elif isinstance(tcp_pose, dict) and 'pos' in tcp_pose:
            x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = tcp_pose['pos']
        elif isinstance(tcp_pose, (list, tuple, np.ndarray)) and len(tcp_pose) >= 6:
            x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = tcp_pose[:6]
        else:
            raise ValueError(f"Unexpected tcp_pose format: {type(tcp_pose)}, length TCP: {len(tcp_pose)}")
        
        # Create rotation matrix
        R_mat = euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg)
        
        # Create translation vector (convert mm to m)
        t_vec = np.array([[x_mm], [y_mm], [z_mm]], dtype=np.float64) 
        
        # Create 4x4 transformation matrix
        T_ee2base = np.concatenate((R_mat, t_vec), axis=1)
        T_ee2base = np.concatenate((T_ee2base, np.array([[0, 0, 0, 1]])), axis=0)

        return T_ee2base

    def screw_down_pos(self, distance_mm=10):
        # Convert local point to robot base frame
        T_ee2base = self.get_robot_transform_matrix(self.robot_real_instance.getcurrent_TCP())
        local_point_homogeneous = np.array([[0, -distance_mm, 0, 1]]).T
        base_point = T_ee2base @ local_point_homogeneous
        x_base, y_base, z_base = base_point[0,0], base_point[1,0], base_point[2,0]
        rx, ry, rz = self.robot_real_instance.getcurrent_TCP()[3:]
        logger.debug(f"Screw down position calculated: {[x_base*1000, y_base*1000, z_base*1000, rx, ry, rz]}")
        return [x_base*1000, y_base*1000, z_base*1000, rx, ry, rz]
    
    def screw_back_pos(self, distance_mm=10):
        # Convert local point to robot base frame
        T_ee2base = self.get_robot_transform_matrix(self.robot_real_instance.getcurrent_TCP())
        local_point_homogeneous = np.array([[0, 0, -distance_mm, 1]]).T
        base_point = T_ee2base @ local_point_homogeneous
        x_base, y_base, z_base = base_point[0,0], base_point[1,0], base_point[2,0]
        rx, ry, rz = self.robot_real_instance.getcurrent_TCP()[3:]
        logger.debug(f"Screw back position calculated: {[x_base*1000, y_base*1000, z_base*1000, rx, ry, rz]}")
        return [x_base*1000, y_base*1000, z_base*1000, rx, ry, rz]
    
    def pick_screw(self, screw_type='M3'):
        # Define pick positions based on screw type
        pick_positions = {
            'M3': [273.62, -390.3, 97.5, 90, 0, 0],
            'M4': [273.62, -390.3, 97.5, 90, 0, 0]
        }
        if screw_type not in pick_positions:
            raise ValueError(f"Unsupported screw type: {screw_type}")
        
        pick_pos = pick_positions[screw_type]
        self.robot_real_instance.move_linear([pick_pos[0], pick_pos[1], pick_pos[2]+50, pick_pos[3], pick_pos[4], pick_pos[5]])
        self.robot_real_instance.rotate_screw()
        time.sleep(0.1)
        self.robot_real_instance.move_linear(pick_pos)
        time.sleep(0.1)
        self.robot_real_instance.move_linear([pick_pos[0], pick_pos[1], pick_pos[2]+50, pick_pos[3], pick_pos[4], pick_pos[5]])
        self.robot_real_instance.stop_screw_rotation()

    def srewing_hole(self, screw_down_distance=10):
        screw_down_pos = self.screw_down_pos(distance_mm=screw_down_distance)
        self.robot_real_instance.move_linear(screw_down_pos)
        time.sleep(0.1)
        # current_tcp = self.robot_real_instance.getcurrent_TCP()
        # self.robot_real_instance.move_linear([current_tcp[0], current_tcp[1], current_tcp[2]+50, current_tcp[3], current_tcp[4], current_tcp[5]])

    def execute_task(self):
        # Main task execution logic
        
        # Step 1: Move to home position
        home_pos = self.config.get('positions', {}).get('home_position', [273.62, -390.3, 200, 90, 0, 0])
        self.robot_real_instance.move_linear(home_pos)

        # Step 2: Pick screw
        # self.pick_screw(screw_type='M3')

        # Step 3: Move to scan position
        scan_pos = self.config.get('positions', {}).get('scan_position', [-225, -800, 360, 78.5, 0, 0])
        self.robot_real_instance.move_linear(scan_pos)

        # Step 4: Detect screw holes with retry logic
        try:
            print("[INFO] Starting screw hole detection with retry logic...")
            real_world_coords = self.screw_hole_detection(max_attempts=50, retry_delay=0.2)
            fine_tuned_coords = self.ready_position(real_world_coords)
            print(f"[SUCCESS] Successfully detected {len(fine_tuned_coords)} screw hole(s)")
        except RuntimeError as e:
            print(f"[ERROR] Screw hole detection failed: {e}")
            print("[INFO] Returning to home position...")
            self.robot_real_instance.move_linear(home_pos)
            raise  # Re-raise the exception to be handled by the caller

        # Step 5: Screw into detected holes
        if fine_tuned_coords:
            print(f"[INFO] Proceeding to screw {len(fine_tuned_coords)} hole(s)")
            # 5.1 Move above each hole
            for i, coord in enumerate(fine_tuned_coords):
                print(f"[INFO] Processing hole {i+1}/{len(fine_tuned_coords)}")
                self.robot_real_instance.move_linear(coord[1])

                # Rotate and screw down
                # self.robot_real_instance.rotate_screw()
                time.sleep(0.1)
                self.robot_real_instance.move_linear(coord[0])
                time.sleep(3)
                # Move to ready_screw position after screwing
                self.robot_real_instance.move_linear(coord[1])
                time.sleep(0.5)
                # self.robot_real_instance.stop_screw_rotation()
                # self.pick_screw(screw_type='M3')
                print(f"[SUCCESS] Completed hole {i+1}/{len(fine_tuned_coords)}")
        else:
            print("[WARNING] No holes to process")
            
        # 5.2 Return to home position
        self.robot_real_instance.move_linear(home_pos)
        print("[SUCCESS] Task execution completed!")

if __name__ == "__main__":
    roi_model_path = '/home/msis/Desktop/Label-Printer/weights/obb/roi/best.pt'
    object_model_path = '/home/msis/Desktop/Label-Printer/weights/obb/object_detection/best.pt'
    camera_intrinsics = None
    config_path = 'config/config.yaml'

    task_planner = TaskPlanner(roi_model_path, object_model_path, camera_intrinsics, config_path)
    
    print("Task Planner initialized successfully!")
    
    
    try:
        while True:
            
            print("Press Enter to execute task, or type 'quit' to exit...")
            user_input = input().strip().lower()
            
            if user_input == 'quit' or user_input == 'exit':
                print("Exiting Task Planner...")
                break
            elif user_input == '' or user_input == 'run':
                print("Executing task...")
                try:
                    task_planner.execute_task()
                    print("Task completed successfully!")
                except Exception as e:
                    print(f"Error during task execution: {e}")
                print("Press Enter to execute task again, or type 'quit' to exit...")
            else:
                print("Unknown command. Press Enter to execute task, or type 'quit' to exit...")
                
    except KeyboardInterrupt:
        print("\nExiting Task Planner...")
    # finally:
    #     # Clean up camera pipeline if needed
    #     if hasattr(task_planner, 'pipeline'):
    #         task_planner.pipeline.stop()
    
