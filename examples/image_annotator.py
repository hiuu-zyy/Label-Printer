#!/usr/bin/env python3
"""
Image Annotation Tool for Label Printer Project

This tool allows you to annotate images with four classes:
- H: Holes (HBB - Horizontal Bounding Box)
- R: ROI (HBB - Horizontal Bounding Box) 
- RF: Radio Frequency (HBB - Horizontal Bounding Box)
- P: Polygon (OBB - Oriented Bounding Box with 4 points)

Usage:
    python examples/image_annotator.py [image_path]

Controls:
    - Press 'h' to switch to Hole annotation mode
    - Press 'r' to switch to ROI annotation mode
    - Press 'f' to switch to Radio Frequency annotation mode
    - Press 'p' to switch to Polygon annotation mode
    - Left click and drag to draw HBB (for H and R classes)
    - Left click 4 points to define polygon (for P class)
    - Press 'z' to undo last annotation
    - Press 's' to save annotations
    - Press 'q' or ESC to quit
    - Press 'c' to clear all annotations
"""

import cv2
import numpy as np
import json
import os
import argparse
from typing import List, Tuple, Dict, Any
import time

class ImageAnnotator:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.original_image = self.image.copy()
        self.current_image = self.image.copy()
        
        # Annotation state
        self.current_class = 'H'  # Default to Hole
        self.annotations = {'H': [], 'R': [], 'RF': [], 'P': []}
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.polygon_points = []
        self.temp_points = []
        
        # Colors for different classes
        self.colors = {
            'H': (0, 255, 0),    # Green for Holes
            'R': (255, 0, 0),    # Blue for ROI
            'RF': (0, 255, 255), # Yellow for Radio Frequency
            'P': (0, 0, 255)     # Red for Polygon
        }
        
        # Window setup
        self.window_name = f"Image Annotator - {os.path.basename(image_path)}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Image Annotator initialized")
        print("Controls:")
        print("  h - Switch to Hole (H) annotation mode")
        print("  r - Switch to Rubber slot (R) annotation mode")
        print("  f - Switch to Rubber Foot (RF) annotation mode") 
        print("  p - Switch to Polygon (P) annotation mode")
        print("  z - Undo last annotation")
        print("  s - Save annotations")
        print("  c - Clear all annotations")
        print("  q/ESC - Quit")
        print(f"Current mode: {self.current_class}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for annotation"""
        if self.current_class in ['H', 'R', 'RF']:
            # Handle HBB (rectangular) annotation
            self._handle_hbb_mouse(event, x, y, flags, param)
        elif self.current_class == 'P':
            # Handle polygon annotation
            self._handle_polygon_mouse(event, x, y, flags, param)

    def _handle_hbb_mouse(self, event, x, y, flags, param):
        """Handle mouse events for horizontal bounding box (H, R, and RF classes)"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Show preview rectangle
                temp_image = self.current_image.copy()
                cv2.rectangle(temp_image, self.start_point, (x, y), 
                            self.colors[self.current_class], 2)
                cv2.imshow(self.window_name, temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Ensure proper rectangle coordinates
                x1, y1 = self.start_point
                x2, y2 = end_point
                
                # Make sure x1,y1 is top-left and x2,y2 is bottom-right
                bbox = [(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))]
                
                # Only add if rectangle has area
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    self.annotations[self.current_class].append(bbox)
                    print(f"Added {self.current_class} annotation: {bbox}")
                    self._update_display()

    def _handle_polygon_mouse(self, event, x, y, flags, param):
        """Handle mouse events for polygon annotation (P class)"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon_points.append((x, y))
            print(f"Polygon point {len(self.polygon_points)}: ({x}, {y})")
            
            # Draw the point
            temp_image = self.current_image.copy()
            
            # Draw existing points
            for i, point in enumerate(self.polygon_points):
                cv2.circle(temp_image, point, 5, self.colors[self.current_class], -1)
                cv2.putText(temp_image, str(i+1), (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[self.current_class], 1)
            
            # Draw lines between points
            if len(self.polygon_points) > 1:
                for i in range(1, len(self.polygon_points)):
                    cv2.line(temp_image, self.polygon_points[i-1], self.polygon_points[i], 
                            self.colors[self.current_class], 2)
            
            # If we have 4 points, close the polygon
            if len(self.polygon_points) == 4:
                cv2.line(temp_image, self.polygon_points[-1], self.polygon_points[0], 
                        self.colors[self.current_class], 2)
                
                # Add the completed polygon to annotations
                self.annotations[self.current_class].append(self.polygon_points.copy())
                print(f"Added {self.current_class} polygon: {self.polygon_points}")
                
                # Reset for next polygon
                self.polygon_points = []
                self._update_display()
            else:
                cv2.imshow(self.window_name, temp_image)

    def _update_display(self):
        """Update the display with all current annotations"""
        self.current_image = self.original_image.copy()
        
        # Draw all HBB annotations (H, R, and RF)
        for class_name in ['H', 'R', 'RF']:
            for bbox in self.annotations[class_name]:
                (x1, y1), (x2, y2) = bbox
                cv2.rectangle(self.current_image, (x1, y1), (x2, y2), 
                            self.colors[class_name], 2)
                # Add class label
                cv2.putText(self.current_image, class_name, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[class_name], 2)
        
        # Draw all polygon annotations (P)
        for polygon in self.annotations['P']:
            points = np.array(polygon, dtype=np.int32)
            cv2.polylines(self.current_image, [points], True, self.colors['P'], 2)
            # Add class label at first point
            cv2.putText(self.current_image, 'P', polygon[0], 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['P'], 2)
        
        # Show current mode
        mode_text = f"Mode: {self.current_class} | Total: H={len(self.annotations['H'])}, R={len(self.annotations['R'])}, RF={len(self.annotations['RF'])}, P={len(self.annotations['P'])}"
        cv2.putText(self.current_image, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.current_image)

    def switch_class(self, new_class: str):
        """Switch annotation class"""
        if new_class in ['H', 'R', 'RF', 'P']:
            self.current_class = new_class
            # Reset any in-progress polygon
            self.polygon_points = []
            self.drawing = False
            print(f"Switched to {new_class} annotation mode")
            self._update_display()

    def undo_last_annotation(self):
        """Remove the last annotation for current class"""
        if self.annotations[self.current_class]:
            removed = self.annotations[self.current_class].pop()
            print(f"Removed last {self.current_class} annotation: {removed}")
            self._update_display()
        else:
            print(f"No {self.current_class} annotations to undo")

    def clear_all_annotations(self):
        """Clear all annotations"""
        self.annotations = {'H': [], 'R': [], 'RF': [], 'P': []}
        self.polygon_points = []
        self.drawing = False
        print("Cleared all annotations")
        self._update_display()

    def save_annotations(self, output_path: str = None):
        """Save annotations to JSON file"""
        if output_path is None:
            base_name = os.path.splitext(self.image_path)[0]
            output_path = f"{base_name}_annotations.json"
        
        # Prepare data in the requested format
        annotation_data = []
        
        for class_name in ['H', 'R', 'RF', 'P']:
            if self.annotations[class_name]:
                class_data = {
                    "class": class_name,
                    "bbox_information": self.annotations[class_name],
                    "number_of_object": len(self.annotations[class_name])
                }
                annotation_data.append(class_data)
        
        # Add metadata
        metadata = {
            "image_path": self.image_path,
            "image_size": {
                "width": self.image.shape[1],
                "height": self.image.shape[0]
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_annotations": sum(len(self.annotations[class_name]) for class_name in ['H', 'R', 'RF', 'P'])
        }
        
        final_data = {
            "metadata": metadata,
            "annotations": annotation_data
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(final_data, f, indent=2)
            print(f"Annotations saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False

    def load_annotations(self, json_path: str):
        """Load annotations from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Clear current annotations
            self.annotations = {'H': [], 'R': [], 'RF': [], 'P': []}
            
            # Load annotations
            if 'annotations' in data:
                for class_data in data['annotations']:
                    class_name = class_data['class']
                    if class_name in self.annotations:
                        self.annotations[class_name] = class_data['bbox_information']
            
            print(f"Loaded annotations from: {json_path}")
            self._update_display()
            return True
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return False

    def run(self):
        """Main annotation loop"""
        print(f"\nStarting annotation for: {self.image_path}")
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('h'):
                self.switch_class('H')
            elif key == ord('r'):
                self.switch_class('R')
            elif key == ord('f'):
                self.switch_class('RF')
            elif key == ord('p'):
                self.switch_class('P')
            elif key == ord('z'):
                self.undo_last_annotation()
            elif key == ord('s'):
                self.save_annotations()
            elif key == ord('c'):
                print("Clear all annotations? Press 'y' to confirm")
                if (cv2.waitKey(0) & 0xFF) == ord('y'):
                    self.clear_all_annotations()
            elif key == ord('l'):
                # Load annotations
                json_path = input("Enter JSON file path to load: ").strip()
                if os.path.exists(json_path):
                    self.load_annotations(json_path)
                else:
                    print(f"File not found: {json_path}")
        
        cv2.destroyAllWindows()
        
        # Ask to save before exiting if there are annotations
        total_annotations = sum(len(self.annotations[class_name]) for class_name in ['H', 'R', 'RF', 'P'])
        if total_annotations > 0:
            print(f"\nYou have {total_annotations} annotations.")
            save_choice = input("Save annotations before exiting? (y/n): ").strip().lower()
            if save_choice == 'y':
                self.save_annotations()

def main():
    parser = argparse.ArgumentParser(description='Image Annotation Tool')
    parser.add_argument('image_path', nargs='?', help='Path to the image file to annotate')
    parser.add_argument('--load-json', help='Path to existing JSON annotations to load')
    
    args = parser.parse_args()
    
    # Get image path
    if args.image_path:
        image_path = args.image_path
    else:
        # If no image provided, look for images in current directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        current_dir = '.'
        image_files = []
        
        for file in os.listdir(current_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if not image_files:
            print("No image files found in current directory.")
            print("Usage: python image_annotator.py <image_path>")
            return
        
        print("Available image files:")
        for i, file in enumerate(image_files):
            print(f"{i+1}. {file}")
        
        try:
            choice = int(input("Select an image (number): ")) - 1
            image_path = image_files[choice]
        except (ValueError, IndexError):
            print("Invalid selection")
            return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    try:
        # Create annotator
        annotator = ImageAnnotator(image_path)
        
        # Load existing annotations if specified
        if args.load_json and os.path.exists(args.load_json):
            annotator.load_annotations(args.load_json)
        
        # Start annotation
        annotator.run()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
