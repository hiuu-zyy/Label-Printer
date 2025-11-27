#!/usr/bin/env python3
"""
Robot TCP Position Recorder

This script connects to the robot and allows users to record the current TCP position
to a YAML file by pressing Enter. Useful for recording waypoints, calibration positions,
or teaching positions.

Features:
- Connect to robot and display current TCP position
- Save TCP positions to YAML file with timestamps
- Interactive recording with Enter key
- Organized output with position names and metadata
- Error handling and connection status

Controls:
- ENTER: Record current TCP position
- 'q' + ENTER: Quit application
- 's' + ENTER: Save positions to file
- 'c' + ENTER: Clear recorded positions

Author: Auto-generated script
Date: November 17, 2025
"""

import os
import sys
import yaml
import time
from datetime import datetime
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from wrapper.wrapper_robot import SimpleRobot
except ImportError:
    print("[ERROR] Could not import SimpleRobot. Make sure the robot module is available.")
    sys.exit(1)


class TCPRecorder:
    """
    Robot TCP position recorder with YAML output.
    
    Records robot TCP positions and saves them to YAML format with metadata.
    """
    
    def __init__(self, robot_ip: str = "192.168.2.36", output_dir: str = "tcp_positions"):
        """
        Initialize the TCP recorder.
        
        Args:
            robot_ip: IP address of the robot controller
            output_dir: Directory to save YAML files
        """
        self.robot_ip = robot_ip
        self.output_dir = output_dir
        self.robot = None
        
        # Position storage
        self.recorded_positions = []
        self.position_counter = 0
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()
        
        print(f"TCP Position Recorder initialized")
        print(f"Session ID: {self.session_id}")
        print(f"Output directory: {self.output_dir}")
    
    def connect_robot(self):
        """Initialize robot connection with retry logic"""
        try:
            print(f"\n[INFO] Connecting to robot at {self.robot_ip}...")
            self.robot = SimpleRobot(ip=self.robot_ip)
            
            # Wait for connection to establish
            time.sleep(2.0)
            print("[INFO] Waiting for robot connection to stabilize...")
            
            # Test connection with retries
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    current_pos = self.robot.getcurrent_TCP()
                    if current_pos is not None:
                        print(f"[SUCCESS] Robot connected successfully")
                        print(f"[INFO] Current TCP position: {current_pos}")
                        return True
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
            return False
    
    def get_current_tcp(self) -> Optional[List[float]]:
        """Get current TCP position from robot"""
        try:
            tcp_pos = self.robot.getcurrent_TCP()
            if tcp_pos is not None:
                # Import numpy at the top to ensure it's available
                try:
                    import numpy as np
                except ImportError:
                    np = None
                
                # Handle different possible return formats
                if isinstance(tcp_pos, tuple) and len(tcp_pos) >= 2:
                    # Format: (success, [x, y, z, rx, ry, rz])
                    position_data = tcp_pos[1]
                    if np and isinstance(position_data, np.ndarray):
                        return position_data[:6].tolist()
                    else:
                        return list(position_data[:6])
                elif np and isinstance(tcp_pos, np.ndarray) and len(tcp_pos) >= 6:
                    # Format: numpy array [x, y, z, rx, ry, rz]
                    return tcp_pos[:6].tolist()
                elif isinstance(tcp_pos, (list, tuple)) and len(tcp_pos) >= 6:
                    # Format: [x, y, z, rx, ry, rz]
                    return list(tcp_pos[:6])
                elif hasattr(tcp_pos, '__len__') and len(tcp_pos) >= 6:
                    # Generic array-like object
                    try:
                        return [float(tcp_pos[i]) for i in range(6)]
                    except (IndexError, ValueError, TypeError) as e:
                        print(f"[ERROR] Could not convert TCP data to float list: {e}")
                        print(f"[DEBUG] TCP data: {tcp_pos}")
                        return None
                else:
                    print(f"[WARNING] Unexpected TCP format: {type(tcp_pos)}")
                    print(f"[DEBUG] TCP length: {len(tcp_pos) if hasattr(tcp_pos, '__len__') else 'unknown'}")
                    print(f"[DEBUG] TCP data: {tcp_pos}")
                    return None
            else:
                print(f"[ERROR] TCP position is None")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to get TCP position: {e}")
            import traceback
            print(f"[DEBUG] Full error: {traceback.format_exc()}")
            return None
    
    def record_position(self, position_name: str = None) -> bool:
        """Record current TCP position"""
        tcp_pos = self.get_current_tcp()
        if tcp_pos is None:
            print(f"[ERROR] Could not record position - failed to get TCP")
            return False
        
        # Generate position name if not provided
        if position_name is None:
            self.position_counter += 1
            position_name = f"position_{self.position_counter:03d}"
        
        # Create position record
        position_record = {
            "name": position_name,
            "tcp_position": {
                "x": round(tcp_pos[0], 3),     # mm
                "y": round(tcp_pos[1], 3),     # mm
                "z": round(tcp_pos[2], 3),     # mm
                "rx": round(tcp_pos[3], 3),    # degrees
                "ry": round(tcp_pos[4], 3),    # degrees
                "rz": round(tcp_pos[5], 3)     # degrees
            },
            "tcp_array": [round(val, 3) for val in tcp_pos],  # For easy copy-paste
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "session_elapsed": time.time() - self.session_start
        }
        
        self.recorded_positions.append(position_record)
        
        print(f"[RECORDED] {position_name}: {position_record['tcp_array']}")
        print(f"[INFO] Total positions recorded: {len(self.recorded_positions)}")
        
        return True
    
    def save_to_yaml(self, filename: str = None) -> str:
        """Save recorded positions to YAML file"""
        if not self.recorded_positions:
            print("[WARNING] No positions recorded to save")
            return None
        
        if filename is None:
            filename = f"tcp_positions_{self.session_id}.yaml"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare YAML data
        yaml_data = {
            "session_info": {
                "session_id": self.session_id,
                "robot_ip": self.robot_ip,
                "start_time": self.session_start,
                "start_datetime": datetime.fromtimestamp(self.session_start).isoformat(),
                "save_time": time.time(),
                "save_datetime": datetime.now().isoformat(),
                "total_positions": len(self.recorded_positions)
            },
            "positions": {}
        }
        
        # Add positions to YAML data
        for pos_record in self.recorded_positions:
            position_name = pos_record["name"]
            yaml_data["positions"][position_name] = {
                "tcp": pos_record["tcp_array"],
                "coordinates": pos_record["tcp_position"],
                "metadata": {
                    "timestamp": pos_record["timestamp"],
                    "datetime": pos_record["datetime"],
                    "session_elapsed_seconds": pos_record["session_elapsed"]
                }
            }
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
            
            print(f"[SAVED] Positions saved to: {filepath}")
            print(f"[INFO] {len(self.recorded_positions)} positions saved")
            return filepath
            
        except Exception as e:
            print(f"[ERROR] Failed to save YAML file: {e}")
            return None
    
    def clear_positions(self):
        """Clear all recorded positions"""
        count = len(self.recorded_positions)
        self.recorded_positions = []
        self.position_counter = 0
        print(f"[CLEARED] Removed {count} recorded positions")
    
    def display_status(self):
        """Display current robot status and recorded positions"""
        print("\n" + "="*60)
        print("ROBOT TCP POSITION RECORDER")
        print("="*60)
        
        # Current TCP position
        current_tcp = self.get_current_tcp()
        if current_tcp:
            print(f"Current TCP: {[round(val, 3) for val in current_tcp]}")
        else:
            print("Current TCP: [ERROR - Could not read position]")
        
        print(f"Recorded positions: {len(self.recorded_positions)}")
        print(f"Session elapsed: {time.time() - self.session_start:.1f}s")
        
        # Show recent positions
        if self.recorded_positions:
            print(f"\nRecent positions:")
            recent_positions = self.recorded_positions[-5:]  # Last 5
            for pos in recent_positions:
                elapsed = pos['session_elapsed']
                print(f"  {pos['name']}: {pos['tcp_array']} (t={elapsed:.1f}s)")
        
        print("\nControls:")
        print("  ENTER       - Record current TCP position")
        print("  'name' + ENTER - Record position with custom name")
        print("  's' + ENTER - Save all positions to YAML")
        print("  'c' + ENTER - Clear all recorded positions")
        print("  'q' + ENTER - Quit application")
        print("="*60 + "\n")
    
    def run_interactive(self):
        """Run interactive recording session"""
        print("\n" + "="*60)
        print("INTERACTIVE TCP POSITION RECORDING")
        print("="*60)
        print("Press ENTER to record current position")
        print("Type 'help' for commands")
        print("="*60)
        
        try:
            while True:
                # Display current status
                self.display_status()
                
                # Get user input
                user_input = input("Command (ENTER=record, 'q'=quit): ").strip()
                
                if user_input.lower() == 'q':
                    print("\nQuitting...")
                    break
                elif user_input.lower() == 's':
                    self.save_to_yaml()
                elif user_input.lower() == 'c':
                    confirm = input("Clear all positions? (y/N): ").strip().lower()
                    if confirm == 'y':
                        self.clear_positions()
                elif user_input.lower() == 'help':
                    continue  # Will show status again
                elif user_input == '':
                    # Record position with auto-generated name
                    self.record_position()
                else:
                    # Record position with custom name
                    position_name = user_input.replace(' ', '_')  # Replace spaces
                    self.record_position(position_name)
                
                print()  # Add spacing
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl+C)")
        
        # Auto-save on exit if positions exist
        if self.recorded_positions:
            print(f"\nAuto-saving {len(self.recorded_positions)} positions before exit...")
            saved_file = self.save_to_yaml()
            if saved_file:
                print(f"Positions saved to: {saved_file}")
    
    def run_simple_mode(self):
        """Run simple mode - just press Enter to record"""
        print("\n" + "="*50)
        print("SIMPLE TCP RECORDING MODE")
        print("="*50)
        print("Press ENTER to record current TCP position")
        print("Press Ctrl+C to quit and save")
        print("="*50 + "\n")
        
        try:
            while True:
                # Show current position
                current_tcp = self.get_current_tcp()
                if current_tcp:
                    print(f"Current TCP: {[round(val, 3) for val in current_tcp]}")
                else:
                    print("Current TCP: [ERROR]")
                
                print(f"Recorded: {len(self.recorded_positions)} positions")
                
                # Wait for Enter
                input("Press ENTER to record this position...")
                
                # Record position
                if self.record_position():
                    print(f"✓ Position recorded successfully!\n")
                else:
                    print(f"✗ Failed to record position\n")
                
        except KeyboardInterrupt:
            print(f"\n[INFO] Recording stopped by user (Ctrl+C)")
        
        # Auto-save on exit
        if self.recorded_positions:
            print(f"Saving {len(self.recorded_positions)} positions...")
            saved_file = self.save_to_yaml()
            if saved_file:
                print(f"Positions saved to: {saved_file}")


def main():
    """Main function"""
    print("Robot TCP Position Recorder")
    print("="*30)
    
    # Configuration
    print("\nSelect recording mode:")
    print("1. Simple mode (just press ENTER to record)")
    print("2. Interactive mode (with commands and naming)")
    
    try:
        mode_choice = input("Select mode (1 or 2): ").strip()
        
        # Initialize recorder
        recorder = TCPRecorder(
            robot_ip="192.168.2.36",  # Update with your robot IP
            output_dir="tcp_positions"
        )
        
        # Connect to robot
        if not recorder.connect_robot():
            print("[ERROR] Could not connect to robot. Exiting.")
            return
        
        # Run selected mode
        if mode_choice == "2":
            recorder.run_interactive()
        else:
            recorder.run_simple_mode()
        
        print("\n[INFO] TCP recording session completed")
        
    except KeyboardInterrupt:
        print(f"\n[INFO] Program interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Program failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()