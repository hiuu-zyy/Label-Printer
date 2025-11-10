#!/usr/bin/env python3
"""
Simple camera test for multi-position capture system.

This script tests just the camera capture functionality without requiring
a robot connection. Useful for debugging camera issues.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multi_position_capture import MultiPositionCapture, create_grid_offsets


def test_camera_only():
    """Test camera capture in simulation mode"""
    print("Camera-Only Test")
    print("=" * 20)
    
    # Simple test configuration
    reference_position = [-225, -800, 360, 78.5, 0, 0]
    delta_offsets = [
        {'dx': 0, 'dy': 0, 'dz': 0, 'drx': 0, 'dry': 0, 'drz': 0},     # Reference
        {'dx': 10, 'dy': 10, 'dz': 0, 'drx': 0, 'dry': 0, 'drz': 0},   # Offset 1
        {'dx': -10, 'dy': 10, 'dz': 0, 'drx': 0, 'dry': 0, 'drz': 0}   # Offset 2
    ]
    
    print(f"Reference position: {reference_position}")
    print(f"Test offsets: {len(delta_offsets)} positions")
    print(f"This will capture {len(delta_offsets)} images (simulated robot positions)")
    
    # Initialize capture system in simulation mode
    capture_system = MultiPositionCapture(
        simulation_mode=True,  # No robot required
        camera_width=1280,
        camera_height=720,
        camera_fps=30,
        output_dir="camera_test_output"
    )
    
    try:
        # Initialize components
        print("\nInitializing camera...")
        capture_system.initialize_robot()  # Will skip robot in simulation mode
        capture_system.initialize_camera()
        
        # Run capture sequence
        results = capture_system.run_capture_sequence(
            reference_position=reference_position,
            delta_offsets=delta_offsets,
            return_to_reference=False  # No robot to return
        )
        
        print(f"\n[SUCCESS] Camera test completed!")
        print(f"Images saved to: {capture_system.session_dir}")
        
        # Show results
        stats = results["statistics"] 
        print(f"\nResults:")
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Successful captures: {stats['successful_captures']}")
        print(f"  Failed captures: {stats['failed_captures']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Camera test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        capture_system.cleanup()


def test_manual_capture():
    """Test single image capture manually"""
    print("Manual Camera Test")
    print("=" * 20)
    
    capture_system = MultiPositionCapture(
        simulation_mode=True,
        output_dir="manual_test_output"
    )
    
    try:
        capture_system.initialize_robot()
        capture_system.initialize_camera()
        
        input("\nPress Enter to capture a test image...")
        
        # Capture single image
        metadata = capture_system.capture_images("manual_test", [0, 0, 0, 0, 0, 0])
        
        if metadata:
            print(f"[SUCCESS] Image captured successfully!")
            print(f"Color image: {os.path.join(capture_system.color_dir, metadata['files']['color'])}")
            print(f"Depth image: {os.path.join(capture_system.depth_dir, metadata['files']['depth'])}")
        else:
            print(f"[ERROR] Image capture failed")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Manual test failed: {e}")
        return 1
    finally:
        capture_system.cleanup()


def main():
    """Main function with test options"""
    print("Multi-Position Capture - Camera Test")
    print("=" * 40)
    print("1. Camera-only test (simulate multiple positions)")
    print("2. Manual single capture test") 
    print("3. Exit")
    
    try:
        choice = input("\nSelect test (1-3): ").strip()
        
        if choice == '1':
            return test_camera_only()
        elif choice == '2':
            return test_manual_capture()
        elif choice == '3':
            print("Exiting...")
            return 0
        else:
            print("Invalid choice.")
            return 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        return 0
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
