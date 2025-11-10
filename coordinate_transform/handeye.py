import numpy as np
import math
import pyrealsense2 as rs
import os
from loguru import logger

# Get the base directory (vision_system_fw/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)

def euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg):
    """Euler intrinsic ZYX: R = Rz(rz) @ Ry(ry) @ Rx(rx)"""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    return Rz(rz) @ Ry(ry) @ Rx(rx)

def get_robot_transform_matrix(tcp_pose):
    """Convert robot TCP pose to 4x4 transformation matrix
    
    Args:
        tcp_pose: Can be:
            - tuple: (success, [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg])
            - dict: {'pos': [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]}
            - list/tuple: [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
    
    Returns:
        T_base2ee: 4x4 transformation matrix
    """
    if isinstance(tcp_pose, tuple) and len(tcp_pose) >= 2:
        position_array = tcp_pose[1]
        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = position_array
    elif isinstance(tcp_pose, dict) and 'pos' in tcp_pose:
        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = tcp_pose['pos']
    elif isinstance(tcp_pose, (list, tuple, np.ndarray)) and len(tcp_pose) >= 6:
        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = tcp_pose[:6]
    else:
        raise ValueError(f"Unexpected tcp_pose format: {type(tcp_pose)}, length TCP: {len(tcp_pose)}")
    
    # Create rotation matrix
    R_mat = euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg)
    
    # Create translation vector (convert mm to m)
    t_vec = np.array([[x_mm], [y_mm], [z_mm]], dtype=np.float64) / 1000.0
    
    # Create 4x4 transformation matrix
    T_base2ee = np.concatenate((R_mat, t_vec), axis=1)
    T_base2ee = np.concatenate((T_base2ee, np.array([[0, 0, 0, 1]])), axis=0)
    
    return T_base2ee

class HandeyeTransformer:
    def __init__(self):
        """
        Initialize with pre-calibrated transformation matrix and camera intrinsics
        
        Args:
            T_cam2gripper_file: Path to .npz file containing T_cam2gripper matrix
                              If None, will look for default calibration file
            intrinsic_matrix_file: Path to .npz file containing intrinsic matrix and distortion coeffs
                                 If None, will look for default intrinsic file
        """
        self.T_cam2gripper = None
        self.intrinsic_matrix = None
        self.dist_coeffs = None
        
        
    
    def load_calibration(self, T_cam2gripper_file):
        """Load pre-calibrated transformation matrix"""
        if not os.path.exists(T_cam2gripper_file):
            raise FileNotFoundError(f"Calibration file not found: {T_cam2gripper_file}")
        
        self.T_cam2gripper = np.load(T_cam2gripper_file)['arr_0']
        logger.info(f"Loaded T_cam2gripper from {T_cam2gripper_file}")
        logger.debug(f"T_cam2gripper shape: {self.T_cam2gripper.shape}")
        logger.debug(f"Translation (mm): {(self.T_cam2gripper[:3, 3] * 1000).tolist()}")
    
    def load_intrinsics(self, intrinsic_matrix_file):
        """Load pre-calibrated camera intrinsics from chessboard calibration"""
        if not os.path.exists(intrinsic_matrix_file):
            logger.warning(f"Intrinsic matrix file not found: {intrinsic_matrix_file}")
            logger.info("Will use provided camera_intrinsics dict instead")
            return
        
        try:
            intrinsic_data = np.load(intrinsic_matrix_file)
            self.intrinsic_matrix = intrinsic_data['arr_0']  # 3x3 matrix
            self.dist_coeffs = intrinsic_data['arr_1']       # distortion coefficients
            
            logger.info(f"Loaded camera intrinsics from {intrinsic_matrix_file}")
            logger.debug(f"Intrinsic matrix:\n{self.intrinsic_matrix}")
            logger.debug(f"Distortion coefficients: {self.dist_coeffs.flatten()}")
            
        except Exception as e:
            logger.warning(f"Failed to load intrinsics from {intrinsic_matrix_file}: {e}")
            logger.info("Will use provided camera_intrinsics dict instead")
            self.intrinsic_matrix = None
            self.dist_coeffs = None

    def rot2d(self, yaw_deg: float) -> np.ndarray:
        """2D rotation matrix for coordinate transformation"""
        t = math.radians(yaw_deg)
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s],
                         [s,  c]], dtype=np.float64)

    def median_depth_on_circle(self, depth_array, center, radius, depth_scale, num_points=16):
        """Estimate median depth (m) using points on a circle around the center"""
        cx, cy = center
        H, W = depth_array.shape[:2]
        depths = []
        
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = int(round(cx + radius * math.cos(theta)))
            y = int(round(cy + radius * math.sin(theta)))
            if 0 <= x < W and 0 <= y < H:
                d = depth_array[y, x]
                if d > 0:
                    depths.append(d * depth_scale)
                    
        if depths:
            return float(np.median(depths))
        logger.warning("No valid depth points found on circle")
        return 0.0


    def get_depth_scale(self, depth_image):
        """Get depth scale from RealSense depth frame or use default"""
        try:
            # If depth_image is a RealSense frame, get the depth scale
            if hasattr(depth_image, 'profile'):
                profile = depth_image.profile
                if hasattr(profile, 'get_device'):
                    device = profile.get_device()
                    depth_sensor = device.first_depth_sensor()
                    return depth_sensor.get_depth_scale()
            # Default depth scale for RealSense cameras (typically 0.001 or 0.0001)
            return 0.001  # 1mm per unit
        except:
            return 0.001  # Default fallback
        
    
    def _order_corners_clockwise(self, pts):
        """Đảm bảo 4 đỉnh theo thứ tự vòng quanh (ổn định tính cạnh)."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)[:4]
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
        order = np.argsort(ang)  # tăng dần góc quanh tâm
        return pts[order]

    def _wrap_to_minus90_90(self, angle_deg):
        """Đưa góc về dải (-90, 90]."""
        a = ((angle_deg + 180.0) % 360.0) - 180.0
        if a <= -90.0:
            a += 180.0
        elif a > 90.0:
            a -= 180.0
        return a

    def angle_from_short_edge_obb_xyxyxyxy(self, obb_coords, assume_image_coords=True):
        """
        Trả về (angle_deg, (p0, p1), ordered_pts)
        - angle_deg: góc của CẠNH NGẮN NHẤT so với +x, dải (-90, 90]
        - (p0, p1): hai đầu mút cạnh ngắn đã chọn
        - ordered_pts: 4 đỉnh đã sắp theo chiều (clockwise)
        """
        pts = np.array(obb_coords, dtype=float).reshape(-1, 2)[:4]
        P = self._order_corners_clockwise(pts)

        edges = []
        for i in range(4):
            j = (i + 1) % 4
            v = P[j] - P[i]
            length = np.linalg.norm(v)
            dx, dy = v[0], v[1]
            # Nếu muốn góc theo hệ toán học (y hướng lên), đổi dấu dy:
            if not assume_image_coords:
                dy = -dy
            angle_raw = math.degrees(math.atan2(dy, dx))  # so với +x
            edges.append((length, angle_raw, v, P[i], P[j]))

        # Hai cạnh ngắn nhất là 2 cạnh đối diện; chọn một cách định deterministically
        min_len = max(e[0] for e in edges)
        eps = 1e-6
        short_edges = [e for e in edges if abs(e[0] - min_len) <= eps]

        # Ưu tiên vector có dx >= 0 để cố định chiều (giảm mơ hồ 180°)
        short_edges.sort(key=lambda e: (e[2][0] < 0))  # False < True

        length, angle_raw, v, p0, p1 = short_edges[0]
        angle_deg = -self._wrap_to_minus90_90(angle_raw)
        return angle_deg, (p0, p1), P

    def transform(self, point, depth_image, camera_intrinsics=None, depth_scale=None, robot_current_tcp=None):
        """
        Transform detection coordinates to robot coordinates using depth information.
        
        Args:
            model_output: Dictionary containing detection results
            depth_image: Depth image array or RealSense depth frame
            camera_intrinsics: Dictionary with camera parameters including depth_scale
            
        Returns:
            Dictionary with transformed coordinates and metadata
        """
        # Load calibration matrix
        try:
            T_cam2gripper_file = "/home/msis/Desktop/Label-Printer/handeye_calibration_data/FinalTransforms/T_cam2gripper_HORAUD.npz"
            intrinsic_matrix_file = "/home/msis/Desktop/Label-Printer/handeye_calibration_data/FinalTransforms/IntrinsicMatrix.npz"
            # T_cam2gripper_file = "T_cam2gripper_HORAUD.npz"
        except Exception as e:
            logger.error(f"Error loading calibration files: {e}")
            # return {"error": str(e)}
        self.load_calibration(T_cam2gripper_file)
        self.load_intrinsics(intrinsic_matrix_file)

        # Convert TCP pose to transformation matrix
        T_base2ee = get_robot_transform_matrix(robot_current_tcp)

        
        coordinates = []
        
        # Get depth image as numpy array
        if hasattr(depth_image, 'get_data'):
            # RealSense frame
            depth_array = np.asanyarray(depth_image.get_data())
            # Use depth scale from camera intrinsics instead of trying to extract from frame
        else:
            # Already numpy array
            depth_array = depth_image
        
        logger.info(f"Using depth scale: {depth_scale}")
        
        # Image center - use actual intrinsics like calibration code  
        H, W = depth_array.shape[:2]
        
        # Use image center like calibration script, not camera intrinsics
        u0, v0 = W / 2.0, H / 2.0
        
        logger.debug(f"Image size: {W}x{H}")
        logger.debug(f"Using image center: ({u0:.1f}, {v0:.1f})")

        # Transform detection coordinates to robot coordinates
        for i, center in enumerate(point):
            try:
                center_x, center_y = center[0], center[1]
                center_x += (center_x - u0) * 0.02 #pixel_x_factor
                center_x = round(center_x, 2)
                center_y += (center_y - v0) * 0.015 #pixel_y_factor
                center_y = round(center_y, 2)
                # Skip if we couldn't extract center coordinates
                if center_x is None or center_y is None:
                    logger.warning(f"Could not extract center coordinates from detection {i}")
                    continue

                logger.debug(f"Detection summary - Center: ({center_x:.1f}, {center_y:.1f}))")

                # Get depth at detection center
                Z_m = self.median_depth_on_circle(
                    depth_array, (center_x, center_y), 
                    radius=20, depth_scale=depth_scale, num_points=16
                )
                logger.debug(f"Estimated depth at center: {Z_m:.4f} m")
                
                # Apply camera-specific coordinate transformations
                transformed_coords = self._pixel_to_world_with_intrinsics(
                    center_x, center_y, Z_m, T_base2ee,  camera_intrinsics
                )
                print("Transform_coordinate:", transformed_coords)
                # Add tilt angle to the transformed coordinates
                transformed_coords = [
                    float(round(transformed_coords[0]*1000, 2)), 
                    float(round(transformed_coords[1]*1000, 2)),
                    float(round(transformed_coords[2]*1000, 2))
                ]

                if transformed_coords is not None:
                    coordinates.append(transformed_coords)
                    logger.debug(f"Added coordinate: {coordinates[-1]}")
                else:
                    logger.warning(f"Transformation resulted in None for detection {i}")
                    
            except Exception as e:
                logger.error(f"Error processing detection {i}: {e}")
                logger.debug(f"Detection type: {type(center)}")
                if hasattr(center, '__dict__'):
                    logger.debug(f"Detection attributes: {list(center.__dict__.keys())}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        logger.info(f"Total coordinates transformed: {len(coordinates)}")
        return coordinates
    
    def _extract_detection_data(self, detection):
        """
        Extract center coordinates, tilt angle, and OBB flag from detection data.
        Returns: (center_x, center_y, tilt_angle, is_obb_detection)
        """
        center_x, center_y = None, None
        tilt_angle = 0.0
        is_obb_detection = False
        
        # Handle YOLO detection objects with tensor attributes
        if hasattr(detection, 'xyxy') and detection.xyxy is not None:
            # YOLO detection with xyxy format (tensor)
            if hasattr(detection.xyxy, 'cpu'):
                bbox_tensor = detection.xyxy.cpu().numpy()
            else:
                bbox_tensor = np.array(detection.xyxy)
            
            # Handle batch dimension if present
            if len(bbox_tensor.shape) > 1:
                bbox = bbox_tensor[0]
            else:
                bbox = bbox_tensor
            
            # Extract coordinates
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

        # Handle dictionary format detections
        elif isinstance(detection, dict):
            logger.debug(f"Detection keys: {list(detection.keys())}")
            
            # Check for OBB coordinates (prioritize xyxyxyxy format)
            if 'obb_xyxyxyxy' in detection:
                is_obb_detection = True
                obb_coords = detection['obb_xyxyxyxy']
                logger.debug(f"Found OBB xyxyxyxy format: {obb_coords}")
                
                # Handle both nested [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] and flat [x1,y1,x2,y2,x3,y3,x4,y4] formats
                if isinstance(obb_coords, (list, tuple, np.ndarray)):
                    obb_array = np.array(obb_coords)
                    
                    # Check if it's nested format
                    if obb_array.ndim == 2 and obb_array.shape[0] == 4 and obb_array.shape[1] == 2:
                        # Nested format: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                        x_coords = obb_array[:, 0]  # Extract x coordinates
                        y_coords = obb_array[:, 1]  # Extract y coordinates
                        center_x = np.mean(x_coords)
                        center_y = np.mean(y_coords)
                        
                        # Flatten for angle calculation
                        obb_flat = obb_array.flatten()
                        logger.debug(f"Nested OBB format detected, flattened: {obb_flat}")
                    elif len(obb_array) >= 8:
                        # Flat format: [x1,y1,x2,y2,x3,y3,x4,y4]
                        obb_flat = obb_array.flatten()
                        x_coords = obb_flat[0::2]
                        y_coords = obb_flat[1::2]
                        center_x = np.mean(x_coords)
                        center_y = np.mean(y_coords)
                        logger.debug(f"Flat OBB format detected: {obb_flat}")
                    else:
                        logger.warning(f"Invalid OBB format: shape={obb_array.shape}, len={len(obb_array)}")
                        center_x, center_y = self._extract_bbox_center(detection)
                        obb_flat = None
                    
                    # Calculate angle from short edge if we have valid OBB data
                    if obb_flat is not None and len(obb_flat) >= 8:
                        try:
                            angle_deg, _, _ = self.angle_from_short_edge_obb_xyxyxyxy(
                                obb_flat, assume_image_coords=True
                            )
                            tilt_angle = angle_deg
                            logger.debug(f"OBB center: ({center_x:.1f}, {center_y:.1f}), angle: {angle_deg:.2f}°")
                        except Exception as e:
                            logger.warning(f"Error calculating OBB angle: {e}")
                            tilt_angle = 0.0
            
            # Check for legacy OBB xywhr format
            elif 'obb_xywhr' in detection:
                is_obb_detection = True
                obb_xywhr = detection['obb_xywhr']
                logger.debug("Found legacy OBB xywhr format")
                
                if isinstance(obb_xywhr, (list, tuple, np.ndarray)) and len(obb_xywhr) >= 5:
                    center_x, center_y = obb_xywhr[0], obb_xywhr[1]
                    tilt_angle = detection.get('tilt_angle', obb_xywhr[4] * 180.0 / 3.141592653589793)
                    logger.debug(f"Legacy OBB center: ({center_x:.1f}, {center_y:.1f}), angle: {tilt_angle:.2f}°")
            
            # Check for explicit tilt_angle (indicates OBB even if coordinates are bbox)
            elif 'tilt_angle' in detection and detection['tilt_angle'] != 0.0:
                is_obb_detection = True
                tilt_angle = detection['tilt_angle']
                logger.debug(f"Found explicit tilt_angle: {tilt_angle}° - treating as OBB")
            
            # Fallback to regular bbox if no OBB data found
            if center_x is None or center_y is None:
                center_x, center_y = self._extract_bbox_center(detection)
        
        return center_x, center_y, tilt_angle, is_obb_detection

    def _extract_bbox_center(self, detection):
        """Extract center coordinates from various bbox formats"""
        center_x, center_y = None, None
        
        # Try different possible bbox key names
        bbox = None
        for bbox_key in ['bbox', 'box', 'xyxy', 'coordinates']:
            if bbox_key in detection:
                bbox = detection[bbox_key]
                logger.debug(f"Found bbox under key '{bbox_key}'")
                break
        
        if bbox is not None:
            try:
                # Handle different bbox formats
                if isinstance(bbox, dict):
                    if 'x' in bbox and 'y' in bbox:
                        if 'width' in bbox and 'height' in bbox:
                            # x, y, width, height format
                            center_x = bbox['x']
                            center_y = bbox['y'] 
                        elif 'x2' in bbox and 'y2' in bbox:
                            # x1, y1, x2, y2 format
                            center_x = (bbox['x'] + bbox['x2']) / 2.0
                            center_y = (bbox['y'] + bbox['y2']) / 2.0
                    elif 'cx' in bbox and 'cy' in bbox:
                        # Explicit center coordinates
                        center_x = bbox['cx']
                        center_y = bbox['cy']
                else:
                    # Handle array/list format
                    if hasattr(bbox, 'cpu'):
                        bbox = bbox.cpu().numpy()
                    bbox = np.array(bbox).flatten()
                    
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                    
                logger.debug(f"Extracted bbox center: ({center_x}, {center_y})")
            except Exception as e:
                logger.error(f"Error processing bbox: {e}")
        
        return center_x, center_y

    def _pixel_to_world_with_intrinsics(self, u, v, depth_m, T_base2ee, camera_intrinsics):
        """Convert pixel to world coordinates using camera intrinsics (calibrated matrix preferred)"""
        # Use calibrated intrinsic matrix if available, otherwise use provided dict
        # if self.intrinsic_matrix is not None:
        #     # Use calibrated intrinsic matrix (3x3) 
        #     fx = self.intrinsic_matrix[0, 0]
        #     fy = self.intrinsic_matrix[1, 1] 
        #     cx = self.intrinsic_matrix[0, 2]
        #     cy = self.intrinsic_matrix[1, 2]
            
        #     print(f"[DEBUG] Using calibrated intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        # else:
        #     # Fall back to provided camera_intrinsics dict
        #     fx = camera_intrinsics['fx']
        #     fy = camera_intrinsics['fy'] 
        #     cx = camera_intrinsics['cx']
        #     cy = camera_intrinsics['cy']
            
        #     print(f"[DEBUG] Using provided intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        # Use the calibrated intrinsic matrix (3x3)
        
        # intrinsic_matrix = self.camera_intrinsics['arr_0']
        # dist_coeffs = self.camera_intrinsics['arr_1']
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        cx = self.intrinsic_matrix[0, 2]
        cy = self.intrinsic_matrix[1, 2]
        # Get width/height from camera_intrinsics dict or fallback to 1280x720
        width = 1280
        height = 720
        if camera_intrinsics is not None:
            width = int(camera_intrinsics.get('width', width))
            height = int(camera_intrinsics.get('height', height))
        # If you have access to the depth array shape, you can also use that
        intrinsics = rs.intrinsics()
        # intrinsics.width = width
        # intrinsics.height = height
        intrinsics.fx = float(fx)
        intrinsics.fy = float(fy)
        intrinsics.ppx = float(cx)
        intrinsics.ppy = float(cy)
        intrinsics.model = rs.distortion.none
        # intrinsics.coeffs = [0, 0, 0, 0, 0]
        if self.dist_coeffs is not None:
            for i in range(min(5, len(self.dist_coeffs.flatten()))):
                intrinsics.coeffs[i] = float(self.dist_coeffs.flatten()[i])
        # Convert pixel to camera coordinates
        P_cam = np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(depth_m))).reshape(3,1)
        logger.debug(f"P_cam (m): {P_cam.flatten()}")
        P_cam_h = np.vstack((P_cam, [1.0]))
        
        # Transform to base frame: P_base = T_base2ee @ T_cam2gripper @ P_cam
        P_base_h = T_base2ee @ self.T_cam2gripper @ P_cam_h
        P_base = P_base_h[:3].flatten()
        
        return P_base