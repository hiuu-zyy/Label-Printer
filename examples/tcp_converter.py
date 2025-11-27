#!/usr/bin/env python3
"""
TCP Position Converter

This script converts TCP positions from YAML format to easily copyable formats.
Takes the recorded TCP positions and outputs them in various formats for easy
copy-paste into code or configuration files.

Features:
- Convert YAML TCP positions to Python list format
- Multiple output formats (Python lists, JSON, CSV, etc.)
- Easy copy-paste format
- Preserve position names and metadata

Usage:
    python examples/tcp_converter.py [yaml_file_path]

Author: Auto-generated script  
Date: November 27, 2025
"""

import os
import sys
import yaml
import json
import csv
from typing import Dict, List, Any

def load_yaml_positions(yaml_file: str) -> Dict:
    """Load TCP positions from YAML file"""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load YAML file: {e}")
        return None

def convert_to_python_lists(data: Dict) -> str:
    """Convert TCP positions to Python list format"""
    output = []
    output.append("# TCP Positions in Python List Format")
    output.append("# Copy and paste these into your code")
    output.append("")
    
    if 'positions' in data:
        for pos_name, pos_data in data['positions'].items():
            if 'tcp' in pos_data:
                tcp_list = pos_data['tcp']
                # Format as Python list with proper spacing
                tcp_str = f"[{', '.join(str(val) for val in tcp_list)}]"
                output.append(f"{pos_name} = {tcp_str}")
        
        output.append("")
        output.append("# All positions as a dictionary:")
        output.append("positions = {")
        for pos_name, pos_data in data['positions'].items():
            if 'tcp' in pos_data:
                tcp_list = pos_data['tcp']
                tcp_str = f"[{', '.join(str(val) for val in tcp_list)}]"
                output.append(f"    '{pos_name}': {tcp_str},")
        output.append("}")
    
    return "\n".join(output)

def convert_to_config_format(data: Dict) -> str:
    """Convert TCP positions to configuration file format"""
    output = []
    output.append("# TCP Positions in Configuration Format")
    output.append("# YAML-style configuration")
    output.append("")
    output.append("positions:")
    
    if 'positions' in data:
        for pos_name, pos_data in data['positions'].items():
            if 'tcp' in pos_data:
                tcp_list = pos_data['tcp']
                output.append(f"  {pos_name}: [{', '.join(str(val) for val in tcp_list)}]")
    
    return "\n".join(output)

def convert_to_csv_format(data: Dict) -> str:
    """Convert TCP positions to CSV format"""
    output = []
    output.append("# TCP Positions in CSV Format")
    output.append("name,x,y,z,rx,ry,rz")
    
    if 'positions' in data:
        for pos_name, pos_data in data['positions'].items():
            if 'tcp' in pos_data:
                tcp_list = pos_data['tcp']
                if len(tcp_list) >= 6:
                    row = f"{pos_name},{','.join(str(val) for val in tcp_list[:6])}"
                    output.append(row)
    
    return "\n".join(output)

def convert_to_robot_script_format(data: Dict) -> str:
    """Convert TCP positions to robot script format"""
    output = []
    output.append("# TCP Positions for Robot Scripts")
    output.append("# Ready to use in robot movement commands")
    output.append("")
    
    if 'positions' in data:
        for pos_name, pos_data in data['positions'].items():
            if 'tcp' in pos_data:
                tcp_list = pos_data['tcp']
                tcp_str = f"[{', '.join(str(val) for val in tcp_list)}]"
                output.append(f"# Move to {pos_name}")
                output.append(f"robot.move_linear({tcp_str})")
                output.append("")
    
    return "\n".join(output)

def convert_to_compact_format(data: Dict) -> str:
    """Convert TCP positions to compact copy-paste format"""
    output = []
    output.append("# Compact TCP Positions - Easy Copy/Paste")
    output.append("")
    
    if 'positions' in data:
        # Just the lists
        output.append("# Position arrays only:")
        for pos_name, pos_data in data['positions'].items():
            if 'tcp' in pos_data:
                tcp_list = pos_data['tcp']
                tcp_str = f"[{', '.join(str(val) for val in tcp_list)}]"
                output.append(tcp_str)
        
        output.append("")
        output.append("# With names:")
        for pos_name, pos_data in data['positions'].items():
            if 'tcp' in pos_data:
                tcp_list = pos_data['tcp']
                tcp_str = f"[{', '.join(str(val) for val in tcp_list)}]"
                output.append(f"# {pos_name}")
                output.append(tcp_str)
    
    return "\n".join(output)

def save_converted_formats(yaml_file: str, data: Dict):
    """Save all converted formats to files"""
    base_name = os.path.splitext(yaml_file)[0]
    
    # Python list format
    python_content = convert_to_python_lists(data)
    python_file = f"{base_name}_python_lists.txt"
    with open(python_file, 'w') as f:
        f.write(python_content)
    print(f"[SAVED] Python lists: {python_file}")
    
    # Configuration format
    config_content = convert_to_config_format(data)
    config_file = f"{base_name}_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    print(f"[SAVED] Config format: {config_file}")
    
    # CSV format
    csv_content = convert_to_csv_format(data)
    csv_file = f"{base_name}_positions.csv"
    with open(csv_file, 'w') as f:
        f.write(csv_content)
    print(f"[SAVED] CSV format: {csv_file}")
    
    # Robot script format
    robot_content = convert_to_robot_script_format(data)
    robot_file = f"{base_name}_robot_script.py"
    with open(robot_file, 'w') as f:
        f.write(robot_content)
    print(f"[SAVED] Robot script: {robot_file}")
    
    # Compact format
    compact_content = convert_to_compact_format(data)
    compact_file = f"{base_name}_compact.txt"
    with open(compact_file, 'w') as f:
        f.write(compact_content)
    print(f"[SAVED] Compact format: {compact_file}")

def main():
    """Main function"""
    print("TCP Position Converter")
    print("="*25)
    
    # Get YAML file path
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        # Look for YAML files in tcp_positions directory
        tcp_dir = "tcp_positions"
        if os.path.exists(tcp_dir):
            yaml_files = [f for f in os.listdir(tcp_dir) if f.endswith('.yaml')]
            if yaml_files:
                print("\nAvailable TCP YAML files:")
                for i, file in enumerate(yaml_files):
                    print(f"  {i+1}. {file}")
                
                try:
                    choice = int(input("Select file number: ")) - 1
                    yaml_file = os.path.join(tcp_dir, yaml_files[choice])
                except (ValueError, IndexError):
                    print("Invalid selection")
                    return
            else:
                print("No YAML files found in tcp_positions directory")
                return
        else:
            print("tcp_positions directory not found")
            print("Usage: python tcp_converter.py <yaml_file_path>")
            return
    
    if not os.path.exists(yaml_file):
        print(f"[ERROR] File not found: {yaml_file}")
        return
    
    print(f"\n[INFO] Converting: {yaml_file}")
    
    # Load YAML data
    data = load_yaml_positions(yaml_file)
    if not data:
        return
    
    # Show session info
    if 'session_info' in data:
        session = data['session_info']
        print(f"[INFO] Session: {session.get('session_id', 'unknown')}")
        print(f"[INFO] Total positions: {session.get('total_positions', 0)}")
    
    # Convert and save all formats
    print(f"\n[INFO] Converting to multiple formats...")
    save_converted_formats(yaml_file, data)
    
    # Display compact format for immediate copy-paste
    print(f"\n" + "="*60)
    print("COMPACT FORMAT - READY TO COPY/PASTE")
    print("="*60)
    print(convert_to_compact_format(data))
    print("="*60)
    
    print(f"\n[SUCCESS] Conversion completed!")
    print(f"[INFO] All formats saved with base name: {os.path.splitext(yaml_file)[0]}")

if __name__ == "__main__":
    main()