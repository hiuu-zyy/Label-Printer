#!/usr/bin/env python3
"""
Time-lapse Image Capture System

This script captures images from a RealSense camera at regular intervals (every 1 second)
without any robot movement. Useful for collecting training data, monitoring processes,
or creating time-lapse sequences.

Features:
- Automatic image capture every 1 second
- RealSense color and depth image capture
- Real-time preview with capture indicators
- Organized output with timestamps
- Keyboard controls for start/stop/settings
- Session statistics and metadata

Controls:
- SPACE: Start/pause capture
- 's': Save single image manually
- 'r': Reset capture counter
- 'q': Quit application
- '+': Increase capture interval
- '-': Decrease capture interval

Author: Auto-generated script
Date: November 17, 2025
"""

import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import json
from datetime import datetime
from typing import Optional, Dict

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TimelapseCapture:
    """
    Time-lapse image capture system for RealSense camera.
    
    Captures images at regular intervals without robot movement.
    Provides real-time preview and organized file output.
    """
    
    def __init__(self, 
                 camera_width: int = 1280,
                 camera_height: int = 720,
                 camera_fps: int = 30,
                 output_dir: str = "timelapse_output",
                 capture_interval: float = 1.0):
        """
        Initialize the time-lapse capture system.
        
        Args:
            camera_width: Camera resolution width
            camera_height: Camera resolution height  
            camera_fps: Camera frame rate
            output_dir: Output directory for captured images
            capture_interval: Time between captures in seconds
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.output_dir = output_dir
        self.capture_interval = capture_interval
        
        # Camera components
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        
        # Capture state
        self.is_capturing = False
        self.last_capture_time = 0
        self.capture_count = 0
        self.session_start_time = None
        
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
        
        print(f"[INFO] Time-lapse capture initialized")
        print(f"[INFO] Session ID: {self.session_id}")
        print(f"[INFO] Output directory: {self.session_dir}")
        print(f"[INFO] Capture interval: {self.capture_interval}s")
        
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
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
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
    
    def capture_image(self, manual: bool = False) -> Optional[Dict]:
        """
        Capture a single image.
        
        Args:
            manual: True if manually triggered, False if automatic
            
        Returns:
            Dictionary with capture metadata, or None if capture failed
        """
        try:
            # Capture multiple frames and use the last one for stability
            for _ in range(3):
                frames = self.pipeline.wait_for_frames()
            
            # Get aligned frames
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print(f"[ERROR] Failed to get valid frames")
                return None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Update capture count
            if not manual:
                self.capture_count += 1
            
            # Generate filenames with counter and timestamp
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # HHMMSS_mmm
            counter = self.capture_count if not manual else "manual"
            base_filename = f"img_{counter:04d}_{timestamp}" if not manual else f"manual_{timestamp}"
            
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
            current_time = time.time()
            session_elapsed = current_time - self.session_start_time if self.session_start_time else 0
            
            metadata = {
                "capture_number": self.capture_count if not manual else "manual",
                "timestamp": current_time,
                "datetime": datetime.now().isoformat(),
                "session_elapsed_seconds": session_elapsed,
                "manual_capture": manual,
                "files": {
                    "color": color_filename,
                    "depth": depth_filename,
                    "depth_colormap": depth_colormap_filename
                },
                "camera": {
                    "width": self.camera_width,
                    "height": self.camera_height,
                    "fps": self.camera_fps,
                    "depth_scale": self.depth_scale,
                    "intrinsics": {
                        "fx": self.intrinsics.fx,
                        "fy": self.intrinsics.fy,
                        "cx": self.intrinsics.ppx,
                        "cy": self.intrinsics.ppy
                    }
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
            
            capture_type = "MANUAL" if manual else "AUTO"
            counter_display = "manual" if manual else f"{self.capture_count:04d}"
            print(f"[{capture_type}] Captured image {counter_display} - {color_filename}")
            
            return metadata
            
        except Exception as e:
            print(f"[ERROR] Failed to capture image: {e}")
            return None
    
    def draw_overlay(self, image):
        """Draw information overlay on the preview image"""
        # Create overlay info
        current_time = time.time()
        session_elapsed = current_time - self.session_start_time if self.session_start_time else 0
        time_since_last = current_time - self.last_capture_time if self.last_capture_time > 0 else 0
        time_to_next = max(0, self.capture_interval - time_since_last) if self.is_capturing else 0
        
        # Status text
        status = "CAPTURING" if self.is_capturing else "PAUSED"
        status_color = (0, 255, 0) if self.is_capturing else (0, 165, 255)  # Green if capturing, orange if paused
        
        info_text = [
            f"Status: {status}",
            f"Captures: {self.capture_count}",
            f"Interval: {self.capture_interval:.1f}s",
            f"Session: {session_elapsed:.0f}s",
            f"Next in: {time_to_next:.1f}s" if self.is_capturing else "Next: PAUSED",
            "",
            "Controls:",
            "  SPACE - Start/Pause",
            "  S - Manual capture", 
            "  R - Reset counter",
            "  +/- - Adjust interval",
            "  Q - Quit"
        ]
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (350, 280), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.75, overlay, 0.25, 0)
        
        # Draw text
        y_offset = 35
        for i, line in enumerate(info_text):
            color = status_color if i == 0 else (255, 255, 255)  # Highlight status line
            font_scale = 0.7 if i == 0 else 0.5
            thickness = 2 if i == 0 else 1
            
            cv2.putText(image, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, thickness, cv2.LINE_AA)
            y_offset += 25 if i < 5 else 20
        
        # Draw progress bar for next capture
        if self.is_capturing and time_since_last > 0:
            progress = min(1.0, time_since_last / self.capture_interval)
            bar_width = 300
            bar_height = 10
            bar_x = 20
            bar_y = image.shape[0] - 30
            
            # Background
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Progress
            progress_width = int(bar_width * progress)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
            # Border
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
        
        return image
    
    def run(self):
        """Main capture loop with real-time preview"""
        print("\n" + "="*50)
        print("Time-lapse Image Capture System")
        print("="*50)
        print("Camera preview window will open...")
        print("Use keyboard controls to start/stop capture")
        print("="*50 + "\n")
        
        try:
            # Initialize camera
            self.initialize_camera()
            
            # Start session timer
            self.session_start_time = time.time()
            
            print("[INFO] Starting preview... Press SPACE to begin capture")
            
            while True:
                # Get frame for preview
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Check if it's time for automatic capture
                current_time = time.time()
                if (self.is_capturing and 
                    current_time - self.last_capture_time >= self.capture_interval):
                    self.capture_image(manual=False)
                    self.last_capture_time = current_time
                
                # Draw overlay
                display_image = self.draw_overlay(color_image.copy())
                
                # Show preview
                cv2.imshow('Time-lapse Capture - RealSense', display_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n[INFO] Quitting...")
                    break
                elif key == ord(' '):  # SPACE - start/pause
                    self.is_capturing = not self.is_capturing
                    if self.is_capturing:
                        print("[INFO] Capture started")
                        self.last_capture_time = current_time
                    else:
                        print("[INFO] Capture paused")
                elif key == ord('s'):  # Manual capture
                    self.capture_image(manual=True)
                elif key == ord('r'):  # Reset counter
                    self.capture_count = 0
                    print("[INFO] Capture counter reset")
                elif key == ord('+') or key == ord('='):  # Increase interval
                    self.capture_interval = min(60.0, self.capture_interval + 0.5)
                    print(f"[INFO] Capture interval: {self.capture_interval:.1f}s")
                elif key == ord('-'):  # Decrease interval
                    self.capture_interval = max(0.1, self.capture_interval - 0.5)
                    print(f"[INFO] Capture interval: {self.capture_interval:.1f}s")
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def save_session_summary(self):
        """Save session summary with statistics"""
        if self.session_start_time is None:
            return
            
        session_duration = time.time() - self.session_start_time
        
        summary = {
            "session_id": self.session_id,
            "start_time": self.session_start_time,
            "start_datetime": datetime.fromtimestamp(self.session_start_time).isoformat(),
            "end_time": time.time(),
            "end_datetime": datetime.now().isoformat(),
            "duration_seconds": session_duration,
            "duration_formatted": f"{session_duration//3600:.0f}h {(session_duration%3600)//60:.0f}m {session_duration%60:.1f}s",
            "statistics": {
                "total_captures": self.capture_count,
                "average_interval": session_duration / max(self.capture_count, 1),
                "capture_interval_setting": self.capture_interval,
                "captures_per_minute": self.capture_count / max(session_duration / 60, 1)
            },
            "camera_settings": {
                "width": self.camera_width,
                "height": self.camera_height,
                "fps": self.camera_fps,
                "depth_scale": self.depth_scale
            },
            "output_directories": {
                "session_dir": self.session_dir,
                "color_dir": self.color_dir,
                "depth_dir": self.depth_dir,
                "depth_colormap_dir": self.depth_colormap_dir,
                "metadata_dir": self.metadata_dir
            }
        }
        
        # Save summary
        summary_path = os.path.join(self.session_dir, "session_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Session ID: {self.session_id}")
        print(f"Duration: {summary['duration_formatted']}")
        print(f"Total captures: {self.capture_count}")
        print(f"Average interval: {summary['statistics']['average_interval']:.1f}s")
        print(f"Captures per minute: {summary['statistics']['captures_per_minute']:.1f}")
        print(f"Output directory: {self.session_dir}")
        print(f"Summary saved: {summary_path}")
        print("="*50)
    
    def cleanup(self):
        """Clean up resources"""
        print("[INFO] Cleaning up...")
        
        try:
            if self.pipeline:
                self.pipeline.stop()
                print("[INFO] Camera pipeline stopped")
        except:
            pass
        
        cv2.destroyAllWindows()
        self.save_session_summary()
        print("[INFO] Cleanup complete")


def main():
    """Main function"""
    print("Time-lapse Image Capture System")
    print("="*35)
    
    # Configuration
    print("\nConfiguration options:")
    print("1. Default (1 second interval)")
    print("2. Fast (0.5 second interval)")  
    print("3. Slow (2 second interval)")
    print("4. Custom interval")
    
    try:
        choice = input("\nSelect configuration (1-4): ").strip()
        
        # Set capture interval based on choice
        if choice == "2":
            interval = 0.5
        elif choice == "3":
            interval = 2.0
        elif choice == "4":
            try:
                interval = float(input("Enter interval in seconds: ").strip())
                if interval <= 0:
                    raise ValueError("Interval must be positive")
            except:
                print("Invalid interval. Using default.")
                interval = 1.0
        else:
            interval = 1.0  # Default
        
        print(f"\n[CONFIG] Capture interval: {interval}s")
        
        # Initialize capture system
        capture_system = TimelapseCapture(
            camera_width=1280,
            camera_height=720,
            camera_fps=30,
            output_dir="timelapse_output",
            capture_interval=interval
        )
        
        # Run capture
        capture_system.run()
        
    except KeyboardInterrupt:
        print(f"\n[INFO] Program interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Program failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()