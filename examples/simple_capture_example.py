#!/usr/bin/env python3
"""
Simple Multi-Position Capture Example

This script demonstrates how to use the multi-position capture system
with predefined configurations from the config file.

Usage:
    python simple_capture_example.py [config_name]

Example:
    python simple_capture_example.py medium_grid
    python simple_capture_example.py x_line_sweep
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multi_position_capture import MultiPositionCapture, create_grid_offsets
import capture_config as config


def create_offsets_from_config(config_data: dict) -> list:
    """
    Create offset list from configuration data.
    
    Args:
        config_data: Configuration dictionary
        
    Returns:
        List of delta offset dictionaries
    """
    pattern = config_data["pattern"]
    params = config_data["parameters"]
    
    if pattern == "grid":
        return create_grid_offsets(
            x_range=params["x_range"],
            y_range=params["y_range"],
            z_range=params["z_range"],
            rotation_offsets=params.get("rotation_offsets")
        )
    
    elif pattern == "circular":
        import math
        offsets = []
        radius = params["radius"]
        num_points = params["num_points"]
        z_offsets = params["z_offsets"]
        
        for z_offset in z_offsets:
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                dx = radius * math.cos(angle)
                dy = radius * math.sin(angle)
                
                offset = {
                    'dx': dx,
                    'dy': dy, 
                    'dz': z_offset,
                    'drx': 0,
                    'dry': 0,
                    'drz': 0
                }
                offsets.append(offset)
        
        return offsets
    
    elif pattern == "spiral":
        import math
        offsets = []
        max_radius = params["max_radius"]
        num_turns = params["num_turns"]
        points_per_turn = params["points_per_turn"]
        z_offset = params["z_offset"]
        
        total_points = num_turns * points_per_turn
        
        for i in range(total_points):
            # Calculate spiral position
            turn_progress = i / points_per_turn
            radius = max_radius * turn_progress / num_turns
            angle = 2 * math.pi * turn_progress
            
            dx = radius * math.cos(angle)
            dy = radius * math.sin(angle)
            
            offset = {
                'dx': dx,
                'dy': dy,
                'dz': z_offset, 
                'drx': 0,
                'dry': 0,
                'drz': 0
            }
            offsets.append(offset)
        
        return offsets
    
    elif pattern == "custom":
        return params["points"]
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def main():
    """Main function"""
    print("Simple Multi-Position Capture Example")
    print("=" * 40)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        # Show available configurations and get user input
        config.list_configurations()
        config_name = input("\nEnter configuration name: ").strip()
    
    try:
        # Get configuration
        capture_config = config.get_configuration(config_name)
        print(f"\nLoaded configuration: {config_name}")
        print(f"Description: {capture_config['description']}")
        
        # Get reference position and create offsets
        reference_position = capture_config["reference_position"]
        delta_offsets = create_offsets_from_config(capture_config)
        
        print(f"Reference position: {reference_position}")
        print(f"Number of offset positions: {len(delta_offsets)}")
        print(f"Total positions: {len(delta_offsets) + 1}")
        
        # Validate safety limits
        if not config.validate_safety_limits(reference_position, delta_offsets):
            print("[ERROR] Configuration violates safety limits. Aborting.")
            return 1
        
        # Confirm execution
        print(f"\nThis will capture {len(delta_offsets) + 1} images at different robot positions.")
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Capture cancelled.")
            return 0
        
        # Initialize capture system
        capture_system = MultiPositionCapture(
            robot_ip=config.ROBOT_IP,
            camera_width=config.CAMERA_WIDTH,
            camera_height=config.CAMERA_HEIGHT,
            camera_fps=config.CAMERA_FPS,
            output_dir=config.OUTPUT_DIR
        )
        
        try:
            # Initialize components
            print("\nInitializing robot and camera...")
            capture_system.initialize_robot()
            capture_system.initialize_camera()
            
            # Run capture sequence
            results = capture_system.run_capture_sequence(
                reference_position=reference_position,
                delta_offsets=delta_offsets,
                return_to_reference=config.MOVEMENT_SETTINGS["return_to_reference"]
            )
            
            print(f"\n[SUCCESS] Capture completed!")
            print(f"Results saved to: {capture_system.session_dir}")
            
            # Show results summary
            stats = results["statistics"]
            print(f"\nCapture Statistics:")
            print(f"  Success rate: {stats['success_rate']:.1f}%")
            print(f"  Successful: {stats['successful_captures']}")
            print(f"  Failed: {stats['failed_captures']}")
            print(f"  Duration: {results['duration_seconds']:.1f}s")
            
            return 0
            
        finally:
            capture_system.cleanup()
    
    except KeyError as e:
        print(f"[ERROR] Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted by user")
        return 0
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
