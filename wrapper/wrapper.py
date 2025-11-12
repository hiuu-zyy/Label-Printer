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
from typing import Dict, List, Tuple, Optional
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
    def __init__(self, roi_model_path, object_model_path, camera_intrinsics, config_path=None, printer_model='XD5-40IIt'):
        """
        Initialize TaskPlanner with specific printer model configuration.
        
        Args:
            roi_model_path: Path to ROI detection model
            object_model_path: Path to object detection model  
            camera_intrinsics: Camera intrinsic parameters
            config_path: Path to configuration file
            printer_model: Specific printer model (XD5-40IIt, XL5-40CT, SLP-DX420, SLP-D220)
        """
        self.printer_model = printer_model
        self.config = {}
        
        # Load configuration file
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Validate printer model exists in config
        if printer_model not in self.config.get('Printer', {}):
            available_models = list(self.config.get('Printer', {}).keys())
            raise ValueError(f"Printer model '{printer_model}' not found in config. Available models: {available_models}")
        
        # Get printer-specific configuration
        printer_config = self.config['Printer'][printer_model]
        
        # Set printer-specific parameters
        self.LP_tilt_angle = printer_config.get('tilt_angle', 45)
        self.screw_depth_mm = printer_config.get('screw_depth_mm', [5, 5, 15, 15])  # Default depths
        self.rotation = [90.0 - self.LP_tilt_angle, 0, 0]
        
        print(f"[INFO] Initialized for printer model: {printer_model}")
        print(f"[INFO] Tilt angle: {self.LP_tilt_angle}°")
        print(f"[INFO] Screw depths: {self.screw_depth_mm} mm")

        # Initialize camera pipeline
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
        
        # Initialize robot instance
        robot_ip = self.config.get('robot', {}).get('ip_address', '192.168.2.36')
        self.robot_real_instance = SimpleRobot(ip=robot_ip)

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
                    points.append((cx, cy))
                    hole_count += 1
            sorted_points = self._sort_detections_clockwise_from_topleft(points)
            logger.debug(f"Sorted hole points: {sorted_points}")
            
            if hole_count > 0:
                print(f"[SUCCESS] Found {hole_count} hole(s) on attempt {attempt}")
                real_world_coordinates = self.handeye_transformer.transform(
                    point=sorted_points,
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

    def _sort_detections_clockwise_from_topleft(self, centers: List[Tuple[float, float]]) -> List[Dict]:
        """
        Sort detections in clockwise order starting from the detection closest to origin (0,0).
        [DEPRECATED] Use _sort_screws_by_yaml_order for drilling order.
        
        Args:
            detections: List of detection dictionaries with bbox containing x, y
            
        Returns:
            Sorted list of detections with closest to origin first, followed by clockwise order
        """
        if not centers or len(centers) <= 1:
            return centers

        import math
        
        # Step 1: Find the detection closest to origin (0, 0)
        def distance_from_origin(center):
            x, y = center
            return math.sqrt(x**2 + y**2)
        
        # Find index of detection closest to origin
        closest_idx = min(range(len(centers)), key=lambda i: distance_from_origin(centers[i]))
        closest_detection = centers[closest_idx]

        # Get coordinates of the starting point (closest to origin)
        start_x, start_y = closest_detection

        logger.debug(f"Starting point closest to origin: ({start_x:.1f}, {start_y:.1f})")
        
        # Step 2: Sort remaining detections clockwise from the starting point
        remaining_detections = [d for i, d in enumerate(centers) if i != closest_idx]
        
        if not remaining_detections:
            return [closest_detection]
        
        # Split remaining detections into positive (x > start_x) and negative (x <= start_x)
        detection_pn = []
        detection_pp = []
        detection_np = []
        detection_nn = []

        for d in remaining_detections:
            x = d[0]
            y = d[1]
            # Treat missing x as start_x (so it goes to negative)
            try:
                if float(x) > float(start_x):
                    if float(y) >= float(start_y):
                        detection_pp.append(d)
                    else:
                        detection_pn.append(d)
                else:
                    if float(y) >= float(start_y):
                        detection_np.append(d)
                    else:
                        detection_nn.append(d)
            except Exception:
                detection_nn.append(d)

        logger.debug(f"Split remaining detections: positive={len(detection_pp)}, negative={len(detection_nn)} (start_x={start_x})")

        # Sort remaining by clockwise angle from starting point
        def get_clockwise_angle(detection):
            x = detection[0]
            y = detection[1]

            # Calculate angle from starting point
            # Use atan2 with negative y to make clockwise (standard math is counter-clockwise)
            angle = math.atan((y - start_y) / (x - start_x))
            
            return angle

        sorted_remaining_1 = sorted(detection_pn, key=get_clockwise_angle)
        sorted_remaining_2 = sorted(detection_pp, key=get_clockwise_angle)
        sorted_remaining_3 = sorted(detection_np, key=get_clockwise_angle)
        sorted_remaining_4 = sorted(detection_nn, key=get_clockwise_angle)

        # Step 3: Combine starting point with sorted remaining detections
        result =  [closest_detection] + sorted_remaining_1 + sorted_remaining_2 + sorted_remaining_3 + sorted_remaining_4

        logger.debug(f"Sorted {len(centers)} detections: starting from origin, then clockwise")

        return result

    def local_movement_y(self, real_world_coordinates, distance_mm):
        # Convert local point to robot base frame
        reference_point = [real_world_coordinates[0], real_world_coordinates[1], real_world_coordinates[2], self.rotation[0], self.rotation[1], self.rotation[2]]
        T_ee2base = self.get_robot_transform_matrix(reference_point)
        local_point_homogeneous = np.array([[0, distance_mm, 0, 1]]).T
        base_point = T_ee2base @ local_point_homogeneous
        base_point = base_point.flatten()
        x_base, y_base, z_base = base_point[0], base_point[1], base_point[2]
        x_base = round(x_base, 2)
        y_base = round(y_base, 2)
        z_base = round(z_base, 2)
        rx, ry, rz = self.rotation
        logger.debug(f"Local movement Y position calculated: {[x_base, y_base, z_base, rx, ry, rz]}")
        return [x_base, y_base, z_base, rx, ry, rz]
    
    def ready_position(self, real_world_coordinates):
        """
        Calculate ready positions for screwing based on printer-specific screw depths.
        
        Args:
            real_world_coordinates: List of detected screw hole coordinates
            
        Returns:
            pair_pos: List of [down_position, up_position] pairs for each screw hole
        """
        # Use printer-specific screw depths
        screw_depths = self.screw_depth_mm
        
        # Ensure we have enough depth values for all holes
        num_holes = len(real_world_coordinates)
        if len(screw_depths) < num_holes:
            # Repeat the last depth value if we don't have enough
            last_depth = screw_depths[-1] if screw_depths else 5
            screw_depths = list(screw_depths) + [last_depth] * (num_holes - len(screw_depths))
            logger.warning(f"Not enough screw depths configured. Extended with last value: {last_depth}mm")
        
        pair_pos = []
        for i, p in enumerate(real_world_coordinates):
            depth_mm = screw_depths[i]
            down = self.local_movement_y(p, -depth_mm)
            logger.debug(f"Screw {i+1} down position (depth: {depth_mm}mm): {down}")
            up = self.local_movement_y(p, 100)
            logger.debug(f"Screw {i+1} up position: {up}")
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
            'M3': [273.62, -390.3, 97, 90, 0, 0],
            'M4': [273.62, -390.3, 97, 90, 0, 0]
        }
        if screw_type not in pick_positions:
            raise ValueError(f"Unsupported screw type: {screw_type}")
        
        pick_pos = pick_positions[screw_type]
        self.robot_real_instance.move_linear([pick_pos[0], pick_pos[1], pick_pos[2]+50, pick_pos[3], pick_pos[4], pick_pos[5]])
        self.robot_real_instance.rotate_screw()
        time.sleep(0.1)
        self.robot_real_instance.move_linear(pick_pos)
        time.sleep(0.5)
        self.robot_real_instance.move_linear([pick_pos[0], pick_pos[1], pick_pos[2]+50, pick_pos[3], pick_pos[4], pick_pos[5]])
        self.robot_real_instance.stop_screw_rotation()

    def srewing_hole(self, screw_down_distance=10):
        screw_down_pos = self.screw_down_pos(distance_mm=screw_down_distance)
        self.robot_real_instance.move_linear(screw_down_pos)
        time.sleep(0.1)
        # current_tcp = self.robot_real_instance.getcurrent_TCP()
        # self.robot_real_instance.move_linear([current_tcp[0], current_tcp[1], current_tcp[2]+50, current_tcp[3], current_tcp[4], current_tcp[5]])

    def execute_task(self):
        """
        Execute the complete screwing task for the configured printer model.
        """
        print(f"[INFO] Starting task execution for printer model: {self.printer_model}")
        
        # Step 1: Move to home position
        home_pos = self.config.get('positions', {}).get('home_position', [273.62, -390.3, 200, 90, 0, 0])
        print(f"[INFO] Moving to home position: {home_pos}")
        self.robot_real_instance.move_linear(home_pos)

        # Step 2: Pick screw
        print("[INFO] Picking screw...")
        self.pick_screw(screw_type='M3')

        # Step 3: Move to scan position
        scan_pos = self.config.get('positions', {}).get('scan_position', [-225, -800, 360, 78.5, 0, 0])
        print(f"[INFO] Moving to scan position: {scan_pos}")
        self.robot_real_instance.move_linear(scan_pos)

        # Step 4: Detect screw holes with retry logic
        try:
            print("[INFO] Starting screw hole detection with retry logic...")
            real_world_coords = self.screw_hole_detection(max_attempts=50, retry_delay=0.2)
            fine_tuned_coords = self.ready_position(real_world_coords)
            print(f"[SUCCESS] Successfully detected {len(fine_tuned_coords)} screw hole(s)")
            print(f"[INFO] Using screw depths: {self.screw_depth_mm[:len(fine_tuned_coords)]} mm")
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
                screw_depth = self.screw_depth_mm[i] if i < len(self.screw_depth_mm) else self.screw_depth_mm[-1]
                print(f"[INFO] Processing hole {i+1}/{len(fine_tuned_coords)} (depth: {screw_depth}mm)")
                self.robot_real_instance.move_linear(coord[1])

                # Rotate and screw down
                self.robot_real_instance.rotate_screw()
                time.sleep(0.1)
                self.robot_real_instance.move_linear(coord[0])
                # Move to ready_screw position after screwing
                self.robot_real_instance.move_linear(coord[1])
                time.sleep(0.5)
                self.robot_real_instance.stop_screw_rotation()
                
                # Pick next screw if not the last hole
                if i < len(fine_tuned_coords) - 1:
                    self.pick_screw(screw_type='M3')
                    
                print(f"[SUCCESS] Completed hole {i+1}/{len(fine_tuned_coords)}")
        else:
            print("[WARNING] No holes to process")
            
        # 5.2 Return to home position
        print("[INFO] Returning to home position...")
        self.robot_real_instance.move_linear(home_pos)
        print(f"[SUCCESS] Task execution completed for {self.printer_model}!")

if __name__ == "__main__":
    roi_model_path = '/home/msis/Desktop/Label-Printer/weights/obb/roi/best.pt'
    object_model_path = '/home/msis/Desktop/Label-Printer/weights/obb/object_detection/best.pt'
    camera_intrinsics = None
    config_path = 'config/config.yaml'

    # Available printer models from config
    available_models = ['XD5-40IIt', 'XL5-40CT', 'SLP-DX420', 'SLP-D220']
    
    print("Available printer models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    
    # Get user selection for printer model
    while True:
        try:
            choice = input(f"\nSelect printer model (1-{len(available_models)}) [default: 1]: ").strip()
            if not choice:
                choice = '1'
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                selected_model = available_models[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nInitializing Task Planner for {selected_model}...")
    
    try:
        task_planner = TaskPlanner(
            roi_model_path=roi_model_path, 
            object_model_path=object_model_path, 
            camera_intrinsics=camera_intrinsics, 
            config_path=config_path,
            printer_model=selected_model
        )
        print("Task Planner initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize Task Planner: {e}")
        exit(1)
    
    try:
        while True:
            print(f"\nCurrent printer model: {selected_model}")
            print("Commands:")
            print("  [Enter] or 'run' - Execute task")
            print("  'change' - Change printer model")
            print("  'quit' or 'exit' - Exit program")
            
            user_input = input(">>> ").strip().lower()
            
            if user_input in ['quit', 'exit']:
                print("Exiting Task Planner...")
                break
            elif user_input == 'change':
                # Change printer model
                print("\nAvailable printer models:")
                for i, model in enumerate(available_models, 1):
                    print(f"{i}. {model}")
                
                try:
                    choice = input(f"Select new printer model (1-{len(available_models)}): ").strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_models):
                        new_model = available_models[choice_idx]
                        if new_model != selected_model:
                            print(f"Reinitializing for {new_model}...")
                            # Cleanup current instance
                            if hasattr(task_planner, 'pipeline'):
                                task_planner.pipeline.stop()
                            # Create new instance
                            task_planner = TaskPlanner(
                                roi_model_path=roi_model_path, 
                                object_model_path=object_model_path, 
                                camera_intrinsics=camera_intrinsics, 
                                config_path=config_path,
                                printer_model=new_model
                            )
                            selected_model = new_model
                            print(f"Successfully switched to {selected_model}")
                        else:
                            print("Same model selected, no change needed")
                    else:
                        print(f"Invalid selection. Please enter 1-{len(available_models)}")
                except ValueError:
                    print("Invalid input. Please enter a number")
                except Exception as e:
                    print(f"Error changing printer model: {e}")
                    
            elif user_input == '' or user_input == 'run':
                print(f"Executing task for {selected_model}...")
                try:
                    task_planner.execute_task()
                    print("Task completed successfully!")
                except Exception as e:
                    print(f"Error during task execution: {e}")
            else:
                print("Unknown command. Use [Enter] to execute, 'change' to switch models, or 'quit' to exit")
                
    except KeyboardInterrupt:
        print("\nExiting Task Planner...")
    finally:
        # Clean up camera pipeline if needed
        try:
            if hasattr(task_planner, 'pipeline'):
                task_planner.pipeline.stop()
        except:
            pass
    
