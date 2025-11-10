#!/usr/bin/env python3
"""
Configuration file for multi-position image capture.

Modify the configurations below to suit your specific capture requirements.
This file contains predefined capture patterns and can be imported by the
main capture script for easy customization.
"""

# Default robot and camera settings
ROBOT_IP = "192.168.2.36"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
OUTPUT_DIR = "capture_output"

# Reference positions for different scenarios
REFERENCE_POSITIONS = {
    "scanning_position": [-225, -800, 360, 78.5, 0, 0],
    "home_position": [273.62, -390.3, 200, 90, 0, 0],
    "calibration_position": [0, -600, 400, 90, 0, 0],
    "workpiece_position": [-200, -750, 300, 90, 0, 0]
}

# Predefined capture configurations
CAPTURE_CONFIGURATIONS = {
    "small_grid": {
        "description": "Small 3x3 grid for quick testing",
        "reference": "scanning_position",
        "pattern": "grid",
        "parameters": {
            "x_range": (-25, 25, 3),    # (min, max, steps) in mm
            "y_range": (-25, 25, 3),
            "z_range": (0, 0, 1),
            "rotation_offsets": None
        }
    },
    
    "medium_grid": {
        "description": "Medium 5x5 grid with Z variation",
        "reference": "scanning_position",
        "pattern": "grid",
        "parameters": {
            "x_range": (-50, 50, 5),
            "y_range": (-50, 50, 5),
            "z_range": (-20, 20, 3),
            "rotation_offsets": None
        }
    },
    
    "large_grid": {
        "description": "Large 7x7 grid for comprehensive coverage",
        "reference": "scanning_position", 
        "pattern": "grid",
        "parameters": {
            "x_range": (-75, 75, 7),
            "y_range": (-75, 75, 7),
            "z_range": (-30, 30, 3),
            "rotation_offsets": None
        }
    },
    
    "x_line_sweep": {
        "description": "Linear sweep along X axis",
        "reference": "scanning_position",
        "pattern": "grid", 
        "parameters": {
            "x_range": (-100, 100, 21),  # 21 points over 200mm
            "y_range": (0, 0, 1),
            "z_range": (0, 0, 1),
            "rotation_offsets": None
        }
    },
    
    "y_line_sweep": {
        "description": "Linear sweep along Y axis",
        "reference": "scanning_position",
        "pattern": "grid",
        "parameters": {
            "x_range": (0, 0, 1),
            "y_range": (-100, 100, 21),  # 21 points over 200mm  
            "z_range": (0, 0, 1),
            "rotation_offsets": None
        }
    },
    
    "z_line_sweep": {
        "description": "Linear sweep along Z axis",
        "reference": "scanning_position",
        "pattern": "grid",
        "parameters": {
            "x_range": (0, 0, 1),
            "y_range": (0, 0, 1),
            "z_range": (-50, 50, 11),    # 11 points over 100mm
            "rotation_offsets": None
        }
    },
    
    "rotation_test": {
        "description": "Test different rotation angles",
        "reference": "scanning_position",
        "pattern": "grid",
        "parameters": {
            "x_range": (0, 0, 1),
            "y_range": (0, 0, 1),
            "z_range": (0, 0, 1),
            "rotation_offsets": [
                (0, 0, 0),      # Reference
                (10, 0, 0), (-10, 0, 0),    # X rotation ±10°
                (0, 10, 0), (0, -10, 0),    # Y rotation ±10°
                (0, 0, 10), (0, 0, -10),    # Z rotation ±10°
                (5, 5, 0), (-5, -5, 0),     # Combined XY
                (0, 5, 5), (0, -5, -5),     # Combined YZ
                (5, 0, 5), (-5, 0, -5)      # Combined XZ
            ]
        }
    },
    
    "circular_pattern": {
        "description": "Circular pattern around reference point",
        "reference": "scanning_position",
        "pattern": "circular",
        "parameters": {
            "radius": 50,           # mm
            "num_points": 8,        # Number of points around circle
            "z_offsets": [0, -20, 20],  # Z levels
            "rotation_towards_center": False
        }
    },
    
    "spiral_pattern": {
        "description": "Spiral pattern outward from center", 
        "reference": "scanning_position",
        "pattern": "spiral",
        "parameters": {
            "max_radius": 60,       # mm
            "num_turns": 3,         # Number of spiral turns
            "points_per_turn": 12,  # Points per turn
            "z_offset": 0           # Single Z level
        }
    },
    
    "custom_points": {
        "description": "Custom defined points",
        "reference": "scanning_position",
        "pattern": "custom",
        "parameters": {
            "points": [
                {"dx": 0, "dy": 0, "dz": 0, "drx": 0, "dry": 0, "drz": 0},      # Reference
                {"dx": 30, "dy": 30, "dz": 0, "drx": 0, "dry": 0, "drz": 0},    # Corner 1
                {"dx": -30, "dy": 30, "dz": 0, "drx": 0, "dry": 0, "drz": 0},   # Corner 2
                {"dx": -30, "dy": -30, "dz": 0, "drx": 0, "dry": 0, "drz": 0},  # Corner 3
                {"dx": 30, "dy": -30, "dz": 0, "drx": 0, "dry": 0, "drz": 0},   # Corner 4
                {"dx": 0, "dy": 0, "dz": 30, "drx": 15, "dry": 0, "drz": 0},    # Above with tilt
                {"dx": 0, "dy": 0, "dz": -30, "drx": -15, "dry": 0, "drz": 0}   # Below with tilt
            ]
        }
    }
}

# Safety limits (absolute values in mm and degrees)
SAFETY_LIMITS = {
    "max_translation": 200,     # Maximum translation from reference in any axis
    "max_rotation": 45,         # Maximum rotation from reference in any axis
    "workspace_bounds": {       # Robot workspace bounds
        "x_min": -500, "x_max": 500,
        "y_min": -1000, "y_max": 100,
        "z_min": 50, "z_max": 600
    }
}

# Movement parameters
MOVEMENT_SETTINGS = {
    "settle_time": 2.0,         # Time to wait after movement (seconds)
    "warmup_frames": 5,         # Number of frames to capture before saving
    "move_speed": "normal",     # Movement speed
    "return_to_reference": True # Return to reference position after sequence
}


def get_configuration(config_name: str) -> dict:
    """
    Get a specific configuration by name.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Configuration dictionary
        
    Raises:
        KeyError: If configuration name not found
    """
    if config_name not in CAPTURE_CONFIGURATIONS:
        available = ", ".join(CAPTURE_CONFIGURATIONS.keys())
        raise KeyError(f"Configuration '{config_name}' not found. Available: {available}")
    
    config = CAPTURE_CONFIGURATIONS[config_name].copy()
    
    # Resolve reference position
    if config["reference"] in REFERENCE_POSITIONS:
        config["reference_position"] = REFERENCE_POSITIONS[config["reference"]]
    else:
        config["reference_position"] = config["reference"]  # Assume it's already a position list
    
    return config


def list_configurations() -> None:
    """Print all available configurations."""
    print("\nAvailable capture configurations:")
    print("=" * 50)
    for name, config in CAPTURE_CONFIGURATIONS.items():
        print(f"{name:20} - {config['description']}")
    print("=" * 50)


def validate_safety_limits(reference_position: list, delta_offsets: list) -> bool:
    """
    Validate that all positions are within safety limits.
    
    Args:
        reference_position: Reference position [x, y, z, rx, ry, rz]
        delta_offsets: List of delta offset dictionaries
        
    Returns:
        True if all positions are safe, False otherwise
    """
    import math
    
    bounds = SAFETY_LIMITS["workspace_bounds"]
    max_trans = SAFETY_LIMITS["max_translation"]
    max_rot = SAFETY_LIMITS["max_rotation"]
    
    # Check reference position
    x, y, z = reference_position[:3]
    if not (bounds["x_min"] <= x <= bounds["x_max"] and
            bounds["y_min"] <= y <= bounds["y_max"] and
            bounds["z_min"] <= z <= bounds["z_max"]):
        print(f"[ERROR] Reference position {reference_position[:3]} outside workspace bounds")
        return False
    
    # Check all offset positions
    for i, delta in enumerate(delta_offsets):
        # Check translation limits
        translation_mag = math.sqrt(
            delta.get('dx', 0)**2 + 
            delta.get('dy', 0)**2 + 
            delta.get('dz', 0)**2
        )
        if translation_mag > max_trans:
            print(f"[ERROR] Offset {i}: Translation magnitude {translation_mag:.1f}mm exceeds limit {max_trans}mm")
            return False
        
        # Check rotation limits
        for axis, key in [('X', 'drx'), ('Y', 'dry'), ('Z', 'drz')]:
            rotation = abs(delta.get(key, 0))
            if rotation > max_rot:
                print(f"[ERROR] Offset {i}: {axis} rotation {rotation:.1f}° exceeds limit {max_rot}°")
                return False
        
        # Check final position bounds
        final_pos = [
            reference_position[0] + delta.get('dx', 0),
            reference_position[1] + delta.get('dy', 0),
            reference_position[2] + delta.get('dz', 0)
        ]
        
        if not (bounds["x_min"] <= final_pos[0] <= bounds["x_max"] and
                bounds["y_min"] <= final_pos[1] <= bounds["y_max"] and
                bounds["z_min"] <= final_pos[2] <= bounds["z_max"]):
            print(f"[ERROR] Offset {i}: Final position {final_pos} outside workspace bounds")
            return False
    
    print(f"[INFO] All {len(delta_offsets)} positions are within safety limits")
    return True


if __name__ == "__main__":
    # Demo: List all configurations
    list_configurations()
    
    # Demo: Show a specific configuration
    print(f"\nExample configuration 'medium_grid':")
    try:
        config = get_configuration("medium_grid")
        print(f"Description: {config['description']}")
        print(f"Reference: {config['reference_position']}")
        print(f"Pattern: {config['pattern']}")
        print(f"Parameters: {config['parameters']}")
    except KeyError as e:
        print(f"Error: {e}")
