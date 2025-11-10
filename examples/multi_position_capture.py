#!/usr/bin/env python3
"""
Multi-Position Image Capture System

This script captures images from a RealSense camera at multiple robot positions
based on a reference point with delta offsets. It moves the robot to each
specified position, captures color and depth images, and saves them with
metadata.

Features:
- Define reference position and delta offsets
- Automated robot movement between positions
- RealSense color and depth image capture
- Organized output with metadata
- Safety checks and error handling
- Progress tracking and logging

Author: Auto-generated script
Date: November 10, 2025
"""

import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from wrapper.wrapper_robot import SimpleRobot
except ImportError:
    print("[ERROR] Could not import SimpleRobot. Make sure the robot module is available.")
    sys.exit(1)


class MultiPositionCapture:
    """
    Multi-position image capture system for RealSense camera with robot movement.
    
    This class handles automated robot movement to predefined positions and
    captures synchronized color/depth images at each location.
    """
    
    def __init__(self, 
                 robot_ip: str = "192.168.2.36",
                 camera_width: int = 1280,
                 camera_height: int = 720,
                 camera_fps: int = 30,
                 output_dir: str = "capture_output",
                 simulation_mode: bool = False):
        """
        Initialize the capture system.
        
        Args:
            robot_ip: IP address of the robot controller
            camera_width: Camera resolution width
            camera_height: Camera resolution height  
            camera_fps: Camera frame rate
            output_dir: Output directory for captured images
            simulation_mode: If True, simulate robot movements without actual robot
        """
        self.robot_ip = robot_ip
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.output_dir = output_dir
        self.simulation_mode = simulation_mode
        
        # Initialize components
        self.robot = None
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        
        # Simulation state
        self.simulated_position = [0, 0, 0, 0, 0, 0]
        
        # Create output directory structure
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{self.session_id}")
        self.color_dir = os.path.join(self.session_dir, "color")
        self.depth_dir = os.path.join(self.session_dir, "depth") 
        self.depth_colormap_dir = os.path.join(self.session_dir, "depth_colormap")
        self.metadata_dir = os.path.join(self.session_dir, "metadata")
        
        # Create directories
        for dir_path in [self.color_dir, self.depth_dir, self.depth_colormap_dir, self.metadata_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Capture statistics
        self.capture_count = 0
        self.success_count = 0
        self.error_count = 0
        
    def initialize_robot(self):
        """Initialize robot connection"""
        if self.simulation_mode:
            print(f"[INFO] Running in simulation mode - no real robot connection")
            print(f"[INFO] Simulated robot initialized at position: {self.simulated_position}")
            return
            
        try:
            print(f"[INFO] Connecting to robot at {self.robot_ip}...")
            self.robot = SimpleRobot(ip=self.robot_ip)
            
            # Wait a moment for connection to establish
            time.sleep(2.0)
            print("[INFO] Waiting for robot connection to stabilize...")
            
            # Test robot connection by getting current position with retries
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    current_pos = self.robot.getcurrent_TCP()
                    if current_pos is not None:
                        print(f"[INFO] Robot connected successfully")
                        print(f"[INFO] Current TCP position: {current_pos}")
                        return
                    else:
                        print(f"[WARNING] Attempt {attempt + 1}: TCP position is None, retrying...")
                        time.sleep(1.0)
                except Exception as retry_error:
                    print(f"[WARNING] Attempt {attempt + 1}: Failed to get TCP position: {retry_error}")
                    if attempt < max_retries - 1:
                        time.sleep(1.0)
                    else:
                        raise
            
            # If we get here, all attempts failed
            raise Exception("Failed to get valid TCP position after all retries")
            
        except Exception as e:
            print(f"[ERROR] Failed to connect to robot: {e}")
            print(f"[HINT] Check that:")
            print(f"  - Robot is powered on and network accessible")
            print(f"  - IP address {self.robot_ip} is correct")
            print(f"  - No other programs are connected to the robot")
            print(f"  - Robot is in proper operating mode")
            print(f"[HINT] To test camera capture only, use simulation_mode=True")
            raise
    
    def initialize_camera(self):
        """Initialize RealSense camera"""
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
            
            print(f"[INFO] Camera initialized successfully")
            print(f"[INFO] Resolution: {self.camera_width}x{self.camera_height}@{self.camera_fps}fps")
            print(f"[INFO] Depth scale: {self.depth_scale}")
            
            # Warmup camera
            print("[INFO] Camera warmup...")
            for _ in range(10):
                self.pipeline.wait_for_frames()
            print("[INFO] Camera ready")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize camera: {e}")
            raise
    
    def move_robot_to_position(self, position: List[float], move_speed: str = "normal") -> bool:
        """
        Move robot to specified position.
        
        Args:
            position: [x, y, z, rx, ry, rz] in mm and degrees
            move_speed: Movement speed ("slow", "normal", "fast")
            
        Returns:
            True if movement successful, False otherwise
        """
        try:
            print(f"[INFO] Moving robot to position: {position}")
            
            if self.simulation_mode:
                # Simulate movement
                print(f"[SIMULATION] Simulating movement to: {position}")
                self.simulated_position = position.copy()
                time.sleep(1.0)  # Simulate movement time
                print(f"[SIMULATION] Robot reached simulated position: {self.simulated_position}")
                return True
            
            # Real robot movement
            # Verify robot is still connected before moving
            try:
                current_pos = self.robot.getcurrent_TCP()
                if current_pos is None:
                    print(f"[ERROR] Lost connection to robot before movement")
                    return False
            except Exception as e:
                print(f"[ERROR] Cannot verify robot position before movement: {e}")
                return False
            
            # Move robot
            self.robot.move_linear(position)
            
            # Wait for robot to settle
            settle_time = 2.0  # seconds
            print(f"[INFO] Waiting {settle_time}s for robot to settle...")
            time.sleep(settle_time)
            
            # Verify position (with error handling)
            try:
                current_pos = self.robot.getcurrent_TCP()
                if current_pos is not None:
                    print(f"[INFO] Robot reached position: {current_pos}")
                else:
                    print(f"[WARNING] Cannot verify final position (TCP is None)")
            except Exception as e:
                print(f"[WARNING] Cannot verify final position: {e}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to move robot to position {position}: {e}")
            return False
    
    def capture_images(self, position_id: str, robot_position: List[float]) -> Optional[Dict]:
        """
        Capture color and depth images at current position.
        
        Args:
            position_id: Unique identifier for this position
            robot_position: Current robot position for metadata
            
        Returns:
            Dictionary with capture metadata, or None if capture failed
        """
        try:
            print(f"[INFO] Capturing images at position {position_id}...")
            
            # Capture multiple frames and use the last one for stability
            for _ in range(5):
                frames = self.pipeline.wait_for_frames()
            
            # Get aligned frames
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print(f"[ERROR] Failed to get valid frames at position {position_id}")
                return None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Generate filenames
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # HHMMSS_mmm
            base_filename = f"pos_{position_id}_{timestamp}"
            
            # Save color image
            color_filename = f"{base_filename}_color.png"
            color_path = os.path.join(self.color_dir, color_filename)
            cv2.imwrite(color_path, color_image)
            
            # Save depth image (raw)
            depth_filename = f"{base_filename}_depth.png"
            depth_path = os.path.join(self.depth_dir, depth_filename)
            cv2.imwrite(depth_path, depth_image)
            
            # Save depth colormap for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            depth_colormap_filename = f"{base_filename}_depth_colormap.png"
            depth_colormap_path = os.path.join(self.depth_colormap_dir, depth_colormap_filename)
            cv2.imwrite(depth_colormap_path, depth_colormap)
            
            # Create metadata
            metadata = {
                "position_id": position_id,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "robot_position": robot_position,
                "files": {
                    "color": color_filename,
                    "depth": depth_filename,
                    "depth_colormap": depth_colormap_filename
                },
                "camera": {
                    "width": self.camera_width,
                    "height": self.camera_height,
                    "fps": self.camera_fps,
                    "depth_scale": self.depth_scale
                },
                "image_stats": {
                    "color_shape": color_image.shape,
                    "depth_shape": depth_image.shape,
                    "depth_min": float(np.min(depth_image[depth_image > 0])) if np.any(depth_image > 0) else 0.0,
                    "depth_max": float(np.max(depth_image)),
                    "depth_mean": float(np.mean(depth_image[depth_image > 0])) if np.any(depth_image > 0) else 0.0
                }
            }
            
            # Save metadata
            metadata_filename = f"{base_filename}_metadata.json"
            metadata_path = os.path.join(self.metadata_dir, metadata_filename)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[SUCCESS] Captured and saved images for position {position_id}")
            print(f"  Color: {color_path}")
            print(f"  Depth: {depth_path}")
            print(f"  Metadata: {metadata_path}")
            
            return metadata
            
        except Exception as e:
            print(f"[ERROR] Failed to capture images at position {position_id}: {e}")
            return None
    
    def calculate_positions(self, 
                          reference_position: List[float],
                          delta_offsets: List[Dict[str, float]]) -> List[Tuple[str, List[float]]]:
        """
        Calculate all positions based on reference and delta offsets.
        
        Args:
            reference_position: [x, y, z, rx, ry, rz] reference position
            delta_offsets: List of dicts with keys: dx, dy, dz, drx, dry, drz
            
        Returns:
            List of (position_id, position) tuples
        """
        positions = []
        
        # Add reference position first
        positions.append(("ref", reference_position.copy()))
        
        # Calculate offset positions
        for i, delta in enumerate(delta_offsets):
            position_id = f"delta_{i+1:02d}"
            
            new_position = [
                reference_position[0] + delta.get('dx', 0),   # x
                reference_position[1] + delta.get('dy', 0),   # y  
                reference_position[2] + delta.get('dz', 0),   # z
                reference_position[3] + delta.get('drx', 0),  # rx
                reference_position[4] + delta.get('dry', 0),  # ry
                reference_position[5] + delta.get('drz', 0)   # rz
            ]
            
            positions.append((position_id, new_position))
        
        return positions
    
    def run_capture_sequence(self, 
                           reference_position: List[float],
                           delta_offsets: List[Dict[str, float]],
                           return_to_reference: bool = True) -> Dict:
        """
        Run the complete capture sequence.
        
        Args:
            reference_position: [x, y, z, rx, ry, rz] reference position
            delta_offsets: List of delta offset dictionaries
            return_to_reference: Whether to return to reference position at the end
            
        Returns:
            Dictionary with capture session results
        """
        print("\n" + "="*60)
        print("Multi-Position Image Capture Sequence")
        print("="*60)
        print(f"Session ID: {self.session_id}")
        print(f"Output directory: {self.session_dir}")
        print(f"Reference position: {reference_position}")
        print(f"Delta offsets: {len(delta_offsets)} positions")
        print("="*60 + "\n")
        
        # Calculate all positions
        positions = self.calculate_positions(reference_position, delta_offsets)
        total_positions = len(positions)
        
        print(f"[INFO] Total positions to capture: {total_positions}")
        for pos_id, pos in positions:
            print(f"  {pos_id}: {pos}")
        
        # Session metadata
        session_metadata = {
            "session_id": self.session_id,
            "start_time": time.time(),
            "start_datetime": datetime.now().isoformat(),
            "reference_position": reference_position,
            "delta_offsets": delta_offsets,
            "total_positions": total_positions,
            "captures": []
        }
        
        try:
            # Execute capture sequence
            for i, (pos_id, position) in enumerate(positions):
                print(f"\n[INFO] === Position {i+1}/{total_positions}: {pos_id} ===")
                
                # Move robot
                if self.move_robot_to_position(position):
                    # Capture images
                    capture_metadata = self.capture_images(pos_id, position)
                    
                    if capture_metadata:
                        session_metadata["captures"].append(capture_metadata)
                        self.success_count += 1
                        print(f"[SUCCESS] Position {pos_id} completed")
                    else:
                        self.error_count += 1
                        print(f"[ERROR] Failed to capture at position {pos_id}")
                else:
                    self.error_count += 1
                    print(f"[ERROR] Failed to move to position {pos_id}")
                
                self.capture_count += 1
                
                # Progress update
                progress = (i + 1) / total_positions * 100
                print(f"[PROGRESS] {progress:.1f}% complete ({i+1}/{total_positions})")
                
                # Small delay between positions
                time.sleep(1.0)
            
            # Return to reference position if requested
            if return_to_reference and len(positions) > 1:
                print(f"\n[INFO] Returning to reference position...")
                self.move_robot_to_position(reference_position)
            
        except KeyboardInterrupt:
            print(f"\n[WARNING] Capture sequence interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error during capture sequence: {e}")
        
        # Finalize session metadata
        session_metadata.update({
            "end_time": time.time(),
            "end_datetime": datetime.now().isoformat(),
            "duration_seconds": time.time() - session_metadata["start_time"],
            "statistics": {
                "total_attempts": self.capture_count,
                "successful_captures": self.success_count,
                "failed_captures": self.error_count,
                "success_rate": self.success_count / max(self.capture_count, 1) * 100
            }
        })
        
        # Save session metadata
        session_file = os.path.join(self.session_dir, "session_metadata.json")
        with open(session_file, 'w') as f:
            json.dump(session_metadata, f, indent=2)
        
        # Print summary
        print(f"\n" + "="*60)
        print("CAPTURE SEQUENCE SUMMARY")
        print("="*60)
        print(f"Session ID: {self.session_id}")
        print(f"Duration: {session_metadata['duration_seconds']:.1f} seconds")
        print(f"Total positions: {total_positions}")
        print(f"Successful captures: {self.success_count}")
        print(f"Failed captures: {self.error_count}")
        print(f"Success rate: {session_metadata['statistics']['success_rate']:.1f}%")
        print(f"Output directory: {self.session_dir}")
        print(f"Session metadata: {session_file}")
        print("="*60)
        
        return session_metadata
    
    def cleanup(self):
        """Clean up resources"""
        print("[INFO] Cleaning up resources...")
        
        try:
            if self.pipeline:
                self.pipeline.stop()
                print("[INFO] Camera pipeline stopped")
        except:
            pass
        
        # Note: Robot connection cleanup is handled by the SimpleRobot class
        print("[INFO] Cleanup complete")


def create_grid_offsets(x_range: Tuple[float, float, int],
                       y_range: Tuple[float, float, int],
                       z_range: Tuple[float, float, int] = (0, 0, 1),
                       rotation_offsets: List[Tuple[float, float, float]] = None) -> List[Dict[str, float]]:
    """
    Create a grid of position offsets.
    
    Args:
        x_range: (min, max, steps) for X axis
        y_range: (min, max, steps) for Y axis  
        z_range: (min, max, steps) for Z axis
        rotation_offsets: List of (drx, dry, drz) rotation offsets
        
    Returns:
        List of delta offset dictionaries
    """
    offsets = []
    
    # Generate position grid
    x_values = np.linspace(x_range[0], x_range[1], x_range[2]) if x_range[2] > 1 else [x_range[0]]
    y_values = np.linspace(y_range[0], y_range[1], y_range[2]) if y_range[2] > 1 else [y_range[0]]
    z_values = np.linspace(z_range[0], z_range[1], z_range[2]) if z_range[2] > 1 else [z_range[0]]
    
    # Default rotation offsets
    if rotation_offsets is None:
        rotation_offsets = [(0, 0, 0)]
    
    # Generate all combinations
    for dx in x_values:
        for dy in y_values:
            for dz in z_values:
                for drx, dry, drz in rotation_offsets:
                    if dx == 0 and dy == 0 and dz == 0 and drx == 0 and dry == 0 and drz == 0:
                        continue  # Skip reference position (will be added separately)
                    
                    offsets.append({
                        'dx': float(dx),
                        'dy': float(dy), 
                        'dz': float(dz),
                        'drx': float(drx),
                        'dry': float(dry),
                        'drz': float(drz)
                    })
    
    return offsets


def main():
    """Main function with example usage"""
    print("Multi-Position Image Capture System")
    print("="*40)
    
    # Example configurations
    examples = {
        "1": {
            "name": "Simple 3x3 grid (±50mm XY)",
            "reference": [-260, -800, 360, 78, 0, 0],
            "offsets": create_grid_offsets(
                x_range=(-50, 50, 3),    # -50 to +50mm in X, 3 steps
                y_range=(-50, 50, 3),    # -50 to +50mm in Y, 3 steps
                z_range=(0, 0, 1)        # No Z offset
            )
        },
        "2": {
            "name": "5x5 grid with Z variation (±30mm XY, ±20mm Z)",
            "reference": [-260, -800, 360, 78, 0, 0],
            "offsets": create_grid_offsets(
                x_range=(-30, 30, 5),    # 5x5 grid
                y_range=(-30, 30, 5),
                z_range=(-20, 20, 3)     # 3 Z levels
            )
        },
        "3": {
            "name": "Linear sweep (X axis, 100mm range)",
            "reference": [-260, -800, 360, 78, 0, 0],
            "offsets": create_grid_offsets(
                x_range=(-50, 50, 11),   # 11 points along X
                y_range=(0, 0, 1),       # No Y variation
                z_range=(0, 0, 1)        # No Z variation
            )
        },
        "4": {
            "name": "Rotation test (±15° around each axis)",
            "reference": [-260, -800, 360, 78, 0, 0],
            "offsets": create_grid_offsets(
                x_range=(0, 0, 1),       # No translation
                y_range=(0, 0, 1),
                z_range=(0, 0, 1),
                rotation_offsets=[
                    (10, 0, 0), (-10, 0, 0),    # X rotation
                    (0, 10, 0), (0, -10, 0),    # Y rotation
                    (0, 0, 10), (0, 0, -10)     # Z rotation
                ]
            )
        },
        "5": {
            "name": "Custom configuration",
            "reference": None,  # Will be set by user input
            "offsets": None     # Will be set by user input
        }
    }
    
    # Show menu
    print("\nAvailable configurations:")
    for key, config in examples.items():
        print(f"  {key}. {config['name']}")
    
    try:
        # Get user choice
        choice = input(f"\nSelect configuration (1-{len(examples)}): ").strip()
        
        if choice not in examples:
            print("Invalid choice. Using default configuration.")
            choice = "1"
        
        config = examples[choice]
        
        # Handle custom configuration
        if choice == "5":
            print("\nCustom configuration:")
            print("Enter reference position [x, y, z, rx, ry, rz]:")
            ref_input = input("Reference (e.g., -225,-800,360,78.5,0,0): ").strip()
            try:
                reference_position = [float(x.strip()) for x in ref_input.split(',')]
                if len(reference_position) != 6:
                    raise ValueError("Must provide exactly 6 values")
            except:
                print("Invalid input. Using default reference position.")
                reference_position = [-225, -800, 360, 78.5, 0, 0]
            
            # Simple grid for custom
            delta_offsets = create_grid_offsets(
                x_range=(-25, 25, 3),
                y_range=(-25, 25, 3),
                z_range=(0, 0, 1)
            )
        else:
            reference_position = config["reference"]
            delta_offsets = config["offsets"]
        
        print(f"\n[CONFIG] {config['name']}")
        print(f"Reference position: {reference_position}")
        print(f"Number of offset positions: {len(delta_offsets)}")
        print(f"Total positions to capture: {len(delta_offsets) + 1}")  # +1 for reference
        
        # Ask about simulation mode
        use_simulation = input(f"\nUse simulation mode (camera only, no robot)? (y/N): ").strip().lower()
        simulation_mode = use_simulation == 'y'
        
        if simulation_mode:
            print("[INFO] Running in simulation mode - camera capture only")
        
        # Confirm before starting
        confirm = input(f"\nProceed with capture? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Capture cancelled.")
            return
        
        # Initialize capture system
        capture_system = MultiPositionCapture(
            robot_ip="192.168.2.36",  # Update with your robot IP
            camera_width=1280,
            camera_height=720,
            camera_fps=30,
            output_dir="capture_output",
            simulation_mode=simulation_mode
        )
        
        try:
            # Initialize components
            capture_system.initialize_robot()
            capture_system.initialize_camera()
            
            # Run capture sequence
            results = capture_system.run_capture_sequence(
                reference_position=reference_position,
                delta_offsets=delta_offsets,
                return_to_reference=True
            )
            
            print(f"\n[SUCCESS] Capture sequence completed successfully!")
            
        finally:
            capture_system.cleanup()
    
    except KeyboardInterrupt:
        print(f"\n[INFO] Program interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Program failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
