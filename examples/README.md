# YOLO Detection Test Examples

This directory contains test scripts and examples for the Label-Printer project.

## Files

### `yolo_detection_test.py`
A comprehensive test script for real-time YOLO object detection using RealSense camera frames.

## Requirements

Before running the test script, ensure you have:

1. **Hardware**: RealSense camera connected to your system
2. **Models**: YOLO model weights in the correct locations:
   - `../weights/obb/roi/best.pt` - ROI detection model
   - `../weights/obb/object_detection/best.pt` - Object detection model
3. **Dependencies**: All required Python packages (see main project requirements)

## Usage

### Basic Usage
```bash
cd examples
python yolo_detection_test.py
```

### Interactive Controls
While the detection window is open:
- **'c'**: Capture and save the current frame with detection results
- **'r'**: Reset all counters (frame count, detection count, etc.)
- **'s'**: Save session summary to file
- **'q'**: Quit the application

## Output

The script creates a `detection_results` directory with:
- **frames/**: Saved color and depth images
- **results/**: JSON files with detection data and metadata
- **session_summary_*.json**: Overall session statistics

### Detection Results Format
```json
{
  "timestamp": "20251108_143022_123",
  "frame_count": 42,
  "detections": [
    {
      "class": "H",
      "center": [640, 360],
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ],
  "camera_intrinsics": {
    "fx": 1234.56,
    "fy": 1234.56,
    "cx": 640,
    "cy": 360,
    "depth_scale": 0.001
  }
}
```

## Features

- **Real-time Detection**: Live YOLO inference on camera feed
- **Two-stage Pipeline**: ROI detection followed by object detection
- **Performance Monitoring**: FPS tracking and frame counting
- **Data Logging**: Automatic saving of frames and detection results
- **Interactive Controls**: Keyboard shortcuts for various actions
- **Error Handling**: Robust error handling and cleanup

## Troubleshooting

### Common Issues

1. **Camera not found**: Ensure RealSense camera is connected and drivers are installed
2. **Model not found**: Check that YOLO model weights are in the correct paths
3. **Import errors**: Make sure you're running from the examples directory or the project has correct paths
4. **Low FPS**: Try reducing camera resolution or frame rate in the script

### Model Paths
If you need to use different model paths, modify the paths at the top of the `main()` function:
```python
roi_model = 'path/to/your/roi/model.pt'
obj_model = 'path/to/your/object/model.pt'
```

## Detection Classes

Based on the VisionWrapper implementation, the system detects:
- **'H'**: Hole detection (blue boxes)
- **'R'**: Rubber detection (cyan boxes)  
- **'x'**: Other objects (magenta boxes)

## Performance Tips

- For better performance on slower systems, reduce camera FPS or resolution
- The script automatically manages GPU/CPU inference based on availability
- Use 'r' to reset counters if running long detection sessions
- Captured frames are saved in PNG format for quality preservation

---

**Note**: This is a test script for development and validation purposes. For production use, consider the main wrapper classes in the parent directory.
