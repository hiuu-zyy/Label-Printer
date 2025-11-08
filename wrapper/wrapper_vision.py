import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class VisionWrapper:
    def __init__(self, roi_model_path='./weights/obb/roi/best.pt', 
                 object_model_path='./weights/obb/object_detection/best.pt',
                 temp_dir='temp'):
        """
        Initialize the VisionWrapper with model paths and configuration.
        
        Args:
            roi_model_path (str): Path to the RoI detection model
            object_model_path (str): Path to the object detection model
            temp_dir (str): Directory for temporary files
        """
        # Load models
        self.roi_model = YOLO(roi_model_path)  # Stage 1: RoI detector
        self.object_model = YOLO(object_model_path)  # Stage 2: Object detector
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    @staticmethod
    def bbox_width_height(pts):
        width = np.linalg.norm(pts[0] - pts[3])
        height = np.linalg.norm(pts[0] - pts[1])
        return width, height

    @staticmethod
    def draw_obb(image, box, label, color=(0, 255, 0), enable_visualization=True):
        """
        Draws an oriented bounding box and label on the image.
        """
        pts = np.array(box).reshape(4, 2)
        pts = pts.astype(int)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        if enable_visualization:
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            cv2.circle(image, (cx, cy), 5, color, -1)
            w, h = VisionWrapper.bbox_width_height(pts)
            w, h = int(w/2), int(h/2)
            cv2.putText(image, label, (cx-0, cy - (h+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return (cx, cy), pts.tolist()

    def run_inference(self, image_or_img_path, enable_vis=True):
        """
        Runs two-stage inference on the input image and returns detections.
        """
        try: 
            img = cv2.imread(image_or_img_path)
        except: 
            pass
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError(f"Invalid image: {image_or_img_path}")

        # Stage 1: Detect RoIs using the first model
        results_roi = self.roi_model(img)
        roi_data = []
        if hasattr(results_roi[0], 'obb') and results_roi[0].obb is not None:
            # Use the xyxyxyxy property to get 8 points per RoI
            roi_data = results_roi[0].obb.xyxyxyxy.cpu().numpy()
        print(f"Detected {len(roi_data)} RoI(s)")

        detections = []

        for i, roi in enumerate([roi_data.flatten()]):
            print(f"RoI {i} points length: {len(roi)}")
            if len(roi) != 8:
                print(f"Skipping RoI {i}: incorrect number of points")
                continue

            # Scale normalized OBB points to image size
            pts = np.array(roi).reshape(4, 2)
            pts_scaled = (pts).astype(np.uint64) # * [img.shape[1], img.shape[0]])

            # Get bounding rectangle from OBB points, clipped to image boundaries
            x_coords = pts_scaled[:, 0]
            y_coords = pts_scaled[:, 1]
            x1, x2 = max(min(x_coords), 0), min(max(x_coords), img.shape[1])
            y1, y2 = max(min(y_coords), 0), min(max(y_coords), img.shape[0])

            crop = img[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                print(f"Skipping empty crop at RoI {i}")
                continue

            # Stage 2: Detect objects within RoI using second model
            results_obj = self.object_model(crop)
            obb_data = []
            obb_classes = []
            if hasattr(results_obj[0], 'obb') and results_obj[0].obb is not None:
                obb_data = results_obj[0].obb.xyxyxyxy.cpu().numpy()
                obb_classes = results_obj[0].obb.cls.cpu().numpy()

            for box, cls_id in zip(obb_data, obb_classes):
                if len(box.flatten()) != 8:
                    continue

                # The detected box is relative to the cropped image size (normalized)
                detected_pts = np.array(box).reshape(4, 2)
                detected_pts_scaled = (detected_pts).astype(np.uint64)

                # Map points back to original image coordinates by adding offsets
                detected_pts_on_img = detected_pts_scaled + [x1, y1]

                label = ['hole', 'rubber', 'other'][int(cls_id)]
                if int(cls_id)==0: 
                    color_ = (120, 160, 255)
                    label  = 'H'
                elif int(cls_id)==1: 
                    color_ = (0, 200, 200)
                    label  = 'R'
                else: 
                    color_ = (255,0,255)
                    label  = 'x'
                center, bbox = self.draw_obb(img, detected_pts_on_img.flatten(), label,color=color_,enable_visualization=enable_vis)
                detections.append({
                    'class': label,
                    'center': center,
                    'bbox': bbox
                })

            # Optional: Draw RoI bounding box for visualization on original image
            if enable_vis:
                self.draw_obb(img, pts_scaled.flatten(), f"RoI {i}", color=(12, 255, 120))
        if enable_vis:
            os.makedirs(self.temp_dir, exist_ok=True)
            output_path = os.path.join(self.temp_dir, 'temp.png')
            cv2.imwrite(output_path, img)
            print(f"Annotated image saved to {output_path}")

        return detections


# Example usage
if __name__ == "__main__":
    # Create an instance of VisionWrapper
    vision_wrapper = VisionWrapper()
    
    image_path = 'data/example_img.png'
    results = vision_wrapper.run_inference(image_path, enable_vis=True)
    for r in results:
        print(r)