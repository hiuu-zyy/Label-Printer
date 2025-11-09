import cv2
import numpy as np
import pyrealsense2 as rs
import rbpodo as rb
# from scipy.spatial.transform import Rotation as R
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import json

# =========================
# CONFIG
# =========================
ROBOT_IP = "192.168.2.36"

CHECKERBOARD = (3, 4)  # Keep original size from calibration2.py
SQUARE_SIZE = 0.0286   # Keep original size from calibration2.py
NUM_SAMPLES = 12

METHOD_MAP = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
}

# Data storage directories
DATA_DIR = "handeye_calibration_data"
RGB_DIR = os.path.join(DATA_DIR, "RGBImgs")
POSE_DIR = os.path.join(DATA_DIR, "T_base2ee")
RESULTS_DIR = os.path.join(DATA_DIR, "FinalTransforms")
CORNERS_DIR = os.path.join(DATA_DIR, "DetectedCorners")

# =========================
# ROBOT CONTROL FUNCTIONS (from calibration2.py)
# =========================
def move_linear(robot, rc, target_point):
    try:
        robot.set_operation_mode(rc, rb.OperationMode.Real)
        rc = rc.error().throw_if_not_empty()
        target_point = np.array(target_point)
        robot.move_l(rc, target_point, 300, 400, rb.ReferenceFrame.Base)
        rc = rc.error().throw_if_not_empty()

        if robot.wait_for_move_started(rc, 0.5).is_success():
            robot.wait_for_move_finished(rc)
        rc = rc.error().throw_if_not_empty()
    finally:
        print('Movement completed')

# def euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg):
#     return R.from_euler('zyx', [rz_deg, ry_deg, rx_deg], degrees=True).as_matrix()

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
    # Euler intrinsic ZYX: R = Rz(rz) @ Ry(ry) @ Rx(rx)
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    return Rz(rz) @ Ry(ry) @ Rx(rx)

def get_robot_transform_matrix(tcp_info):
    """Convert robot TCP info to 4x4 transformation matrix"""
    if isinstance(tcp_info, tuple) and len(tcp_info) >= 2:
        position_array = tcp_info[1]
        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = position_array
    elif isinstance(tcp_info, dict) and 'pos' in tcp_info:
        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = tcp_info['pos']
    elif isinstance(tcp_info, (list, tuple)) and len(tcp_info) >= 6:
        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = tcp_info[:6]
    else:
        raise ValueError(f"Unexpected tcp_info format: {type(tcp_info)}")
    
    # Create rotation matrix
    R_mat = euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg)
    
    # Create translation vector (convert mm to m)
    t_vec = np.array([[x_mm], [y_mm], [z_mm]], dtype=np.float64) / 1000.0
    
    # Create 4x4 transformation matrix
    T_base2ee = np.concatenate((R_mat, t_vec), axis=1)
    T_base2ee = np.concatenate((T_base2ee, np.array([[0, 0, 0, 1]])), axis=0)
    
    return T_base2ee

# =========================
# ENHANCED CALIBRATION CLASS (based on hand_eye_github.py)
# =========================
class RobotHandEyeCalibration:
    """
    Enhanced hand-eye calibration class that combines robot control with robust calibration algorithms
    """
    def __init__(self, robot_ip=ROBOT_IP, pattern_size=CHECKERBOARD, square_size=SQUARE_SIZE, 
                 show_project_error=True, show_corners=True, collect_new_data=True, camera_config_file=None):
        
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.robot_ip = robot_ip
        
        # Load camera configuration
        self.camera_config = self._load_camera_config(camera_config_file)
        
        # Create directories
        for directory in [DATA_DIR, RGB_DIR, POSE_DIR, RESULTS_DIR, CORNERS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize robot connection
        self.robot = None
        self.rc = None
        self._connect_robot()
        
        # Initialize camera
        self.pipeline = None
        self.profile = None
        self._init_camera()
        
        if collect_new_data:
            self._collect_calibration_data()
        
        # Load collected data
        self._load_data()
        
        # Perform calibration
        self._perform_calibration(show_project_error, show_corners)
    
    def _connect_robot(self):
        """Initialize robot connection"""
        try:
            print(f"[INFO] Connecting to robot at {self.robot_ip}...")
            self.robot = rb.Cobot(self.robot_ip)
            self.rc = rb.ResponseCollector()
            print("[INFO] Robot connected successfully")
        except Exception as e:
            print(f"[ERROR] Robot connection failed: {e}")
            raise
    
    def _load_camera_config(self, config_file):
        """Load camera configuration from RealSense SDK JSON file or custom format"""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"[INFO] Loaded camera config from {config_file}")
                
                # Check if it's RealSense SDK format (has 'parameters' and 'viewer' keys)
                if 'parameters' in config and 'viewer' in config:
                    print("[INFO] Detected RealSense SDK configuration format")
                    return config
                else:
                    print("[INFO] Detected custom configuration format")
                    return config
                    
            except Exception as e:
                print(f"[WARN] Failed to load config file {config_file}: {e}")
                print("[INFO] Using default camera configuration")
                return None
        else:
            print("[INFO] No camera configuration file provided")
            return None
    
    def _init_camera(self):
        """Initialize RealSense camera with SDK configuration"""
        print("[INFO] Initializing RealSense camera...")
        # Choose camera by serial id if specified in config
        self.pipeline = rs.pipeline()
        config = rs.config()
        serial_id = "218622274863" 
        config.enable_device(serial_id)
        # default serial id
        # if self.camera_config and 'viewer' in self.camera_config:
        #     serial_id = self.camera_config['viewer'].get('serial-number', None)
        # if serial_id:
        #     ctx = rs.context()
        #     found = False
        #     for dev in ctx.query_devices():
        #         print(f"Found device: {dev.get_info(rs.camera_info.name)} - SN: {dev.get_info(rs.camera_info.serial_number)}")
        #         if dev.get_info(rs.camera_info.serial_number) == serial_id:
        #             found = True
        #             break
        #         if not found:
        #             raise RuntimeError(f"Camera with serial number {serial_id} not found.")
        #     config.enable_device(str(serial_id))
        # config = rs.config()
        
        # Get stream configuration from SDK config if available
        width, height, fps = 1280, 720, 30  # defaults
        if self.camera_config and 'viewer' in self.camera_config:
            viewer = self.camera_config['viewer']
            width = int(viewer.get('stream-width', 1280))
            height = int(viewer.get('stream-height', 720))
            fps = int(viewer.get('stream-fps', 30))
            
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        try:
            self.profile = self.pipeline.start(config)
            device = self.profile.get_device()
            print(f"[INFO] Connected to {device.get_info(rs.camera_info.name)}")
            
            # Apply SDK configuration if available
            if self.camera_config and 'parameters' in self.camera_config:
                self._apply_sdk_camera_settings(device)
            
            # Keep an align object and color intrinsics for later use
            self.align = rs.align(rs.stream.color)
            self.color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.color_intrinsics = self.color_stream.get_intrinsics()
            
            print(f"[INFO] Camera initialized: {width}x{height} @ {fps}fps")
        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            raise
    
    def _apply_sdk_camera_settings(self, device):
        """Apply RealSense SDK camera settings"""
        params = self.camera_config.get('parameters', {})
        
        try:
            # Get sensors
            color_sensor = None
            depth_sensor = None
            
            for sensor in device.sensors:
                if sensor.is_color_sensor():
                    color_sensor = sensor
                elif sensor.is_depth_sensor():
                    depth_sensor = sensor
            
            # Apply color sensor settings
            if color_sensor:
                print("[INFO] Applying color sensor settings...")
                
                # Auto exposure
                if 'controls-autoexposure-auto' in params:
                    auto_exp = params['controls-autoexposure-auto'].lower() == 'true'
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1 if auto_exp else 0)
                    print(f"[INFO] Auto exposure: {'enabled' if auto_exp else 'disabled'}")
                    
                    # Manual exposure (only if auto exposure is disabled)
                    if not auto_exp and 'controls-autoexposure-manual' in params:
                        exposure = float(params['controls-autoexposure-manual'])
                        color_sensor.set_option(rs.option.exposure, exposure)
                        print(f"[INFO] Manual exposure set to: {exposure}")
                
                # Gain
                if 'controls-depth-gain' in params:
                    gain = float(params['controls-depth-gain'])
                    color_sensor.set_option(rs.option.gain, gain)
                    print(f"[INFO] Gain set to: {gain}")
                
                # Auto white balance
                if 'controls-depth-white-balance-auto' in params:
                    auto_wb = params['controls-depth-white-balance-auto'].lower() == 'true'
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 1 if auto_wb else 0)
                    print(f"[INFO] Auto white balance: {'enabled' if auto_wb else 'disabled'}")
            
            # Apply depth sensor settings
            if depth_sensor:
                print("[INFO] Applying depth sensor settings...")
                
                # Apply stereo module parameters
                depth_params = {k: v for k, v in params.items() if k.startswith('param-')}
                for param_name, param_value in depth_params.items():
                    try:
                        # Remove 'param-' prefix to get the actual parameter name
                        actual_param = param_name[6:]  # Remove 'param-'
                        
                        # Convert common parameter names to RealSense options
                        param_mapping = {
                            'depthunits': rs.option.depth_units,
                            'depthclampmin': rs.option.min_distance,
                            'depthclampmax': rs.option.max_distance,
                        }
                        
                        if actual_param in param_mapping:
                            rs_option = param_mapping[actual_param]
                            value = float(param_value)
                            depth_sensor.set_option(rs_option, value)
                            print(f"[INFO] Set {actual_param}: {value}")
                    except Exception as e:
                        # Skip parameters that can't be set
                        pass
                        
            print("[INFO] SDK camera settings applied successfully")
            
        except Exception as e:
            print(f"[WARN] Failed to apply some SDK camera settings: {e}")
    
    def _collect_calibration_data(self):
        """Collect calibration images and robot poses"""
        print(f"[INFO] Collecting {NUM_SAMPLES} calibration samples...")
        
        # Predefined calibration points
        calibration_points = [
            [ 302.   , -110.   ,  413.493,   90.   ,   -0.   ,  -90.   ],
            [279.843, -93.775, 565.41 ,  89.989,  -0.868, -90.128],
            [457.849, -95.536, 337.216,  55.481,  -2.727, -88.195],
            [575.   , -74.908, 434.023,  54.255,   3.52 , -96.983],
            [308.828, 107.006, 320.169,  88.75 , -36.193, -87.153],
            [313.058, 233.461, 465.298,  91.645, -32.746, -87.22 ],
            [ 176.798,    3.761,  432.338,  111.31 ,   -7.172, -123.007],
            [ 265.07 , -285.826,  300.587,   86.762,   43.306, -101.433],
            [ 303.411, -442.338,  466.986,   86.178,   39.774,  -96.713],
            [ 471.47 , -314.96 ,  415.681,   64.933,   32.489,  -97.29 ],
            [485.125, 134.512, 437.996,  59.881,  -3.445, -51.308],
            [474.072,  85.086, 350.991,  53.243,  -3.184, -53.807]
        ]
        
        align_to_color = rs.align(rs.stream.color)
        
        # Warm up camera
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        sample_count = 0
        for i, point in enumerate(calibration_points[:NUM_SAMPLES]):
            print(f"\n=== Sample {i+1}/{NUM_SAMPLES} ===")
            
            # Move robot to calibration point
            try:
                # move_linear(self.robot, self.rc, point)
                input(f"Press Enter when robot is stable at position {i+1}...")
                
                # Get robot pose
                tcp_info = self.robot.get_tcp_info(self.rc)
                print(f"[INFO] Robot TCP info: {tcp_info}")
                T_base2ee = get_robot_transform_matrix(tcp_info)
                
                # Capture image
                frames = self.pipeline.wait_for_frames()
                aligned = align_to_color.process(frames)
                color = aligned.get_color_frame()
                
                if not color:
                    print("[WARN] Failed to capture color frame")
                    continue
                
                image = np.asanyarray(color.get_data())
                
                # Save image and pose
                img_filename = os.path.join(RGB_DIR, f"img_{sample_count:03d}.png")
                pose_filename = os.path.join(POSE_DIR, f"pose_{sample_count:03d}.npz")
                
                cv2.imwrite(img_filename, image)
                np.savez(pose_filename, T_base2ee)
                
                print(f"[INFO] Saved sample {sample_count}: {img_filename}")
                sample_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to collect sample {i+1}: {e}")
                continue
        
        print(f"[INFO] Collected {sample_count} samples successfully")
    
    def _load_data(self):
        """Load images and robot poses from saved data"""
        print("[INFO] Loading calibration data...")
        
        # Load images
        image_files = sorted(glob.glob(os.path.join(RGB_DIR, '*.png')))
        self.images = [cv2.imread(f) for f in image_files]
        self.images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.images]
        
        # Load robot poses
        pose_files = sorted(glob.glob(os.path.join(POSE_DIR, '*.npz')))
        self.T_base2ee_list = [np.load(f)['arr_0'] for f in pose_files]
        
        print(f"[INFO] Loaded {len(self.images)} images and {len(self.T_base2ee_list)} poses")
    
    def _perform_calibration(self, show_project_error=True, show_corners=True):
        """Perform the complete calibration process using camera calibration from chessboard"""
        # Find chessboard corners
        self.chessboard_corners, self.valid_indices = self._find_chessboard_corners(show_corners)
        if len(self.chessboard_corners) == 0:
            raise RuntimeError("No valid chessboard detections. Cannot proceed.")
        
        # Perform camera calibration from chessboard instead of using RealSense intrinsics
        self._calibrate_camera_from_chessboard(show_project_error)
        
        # Filter transforms for valid images only
        self.T_base2ee_list = [self.T_base2ee_list[i] for i in self.valid_indices]
        # Save intrinsic matrix and distortion
        np.savez(os.path.join(RESULTS_DIR, "IntrinsicMatrix.npz"), self.intrinsic_matrix, self.dist_coeffs)
        # Calculate camera poses relative to target
        self.R_target2cam, self.t_target2cam = self._compute_camera_poses()
        # Prepare data for hand-eye calibration
        self._prepare_hand_eye_data()
        # Perform hand-eye calibration with multiple methods
        self._solve_hand_eye_calibration()

    def _calibrate_camera_from_chessboard(self, show_project_error=False):
        """Perform camera calibration using chessboard patterns"""
        print("[INFO] Performing camera calibration from chessboard patterns...")
        
        # Prepare object points
        objpoints = []
        for _ in range(len(self.valid_indices)):
            objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2) * self.square_size
            objpoints.append(objp)
        
        # Get image size from first valid image
        img_size = self.images[self.valid_indices[0]].shape[:2][::-1]  # (width, height)
        
        # Camera calibration
        print(f"[INFO] Calibrating camera with {len(objpoints)} image pairs...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, self.chessboard_corners, img_size, None, None
        )
        
        if not ret:
            raise RuntimeError("Camera calibration failed!")
        
        # Store results
        self.intrinsic_matrix = mtx
        self.dist_coeffs = dist
        
        # Calculate reprojection error
        reprojection_error = self._calculate_reprojection_error(
            objpoints, self.chessboard_corners, rvecs, tvecs, mtx, dist, show_project_error
        )
        
        print(f"[INFO] Camera calibration completed successfully")
        print(f"[INFO] Reprojection error: {reprojection_error:.4f} pixels")
        print(f"[INFO] Camera matrix from chessboard calibration:")
        print(mtx)
        print(f"[INFO] Distortion coefficients: {dist.flatten()}")
        
        return mtx, dist, reprojection_error
    
    def _calculate_reprojection_error(self, objpoints, imgpoints, rvecs, tvecs, mtx, dist, show_plot=False):
        """Calculate reprojection error for camera calibration validation"""
        total_error = 0
        errors = []
        
        for i in range(len(objpoints)):
            imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints_proj = imgpoints_proj.reshape(-1, 1, 2)
            error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
            errors.append(error)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        if show_plot:
            plt.figure(figsize=(12, 6))
            img_indices = range(1, len(errors) + 1)
            plt.bar(img_indices, errors)
            plt.axhline(y=mean_error, color='r', linestyle='--', label=f'Mean Error: {mean_error:.4f}')
            plt.xlabel('Image Index')
            plt.ylabel('Reprojection Error (pixels)')
            plt.title('Reprojection Error for Each Image (Camera Calibration)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(RESULTS_DIR, 'CameraCalibrationReprojectionError.png'))
            plt.show()
        
        return mean_error

    def _set_intrinsics_from_realsense(self):
        """Fetch RealSense internal color camera intrinsics and set intrinsic_matrix/distortion."""
        try:
            # use the color intrinsics captured at init
            intr = self.color_intrinsics
            fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
            self.intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            # RealSense provides up to 5 distortion coeffs for Brown model (k1,k2,p1,p2,k3)
            self.dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64).reshape(1, -1)
            print("[INFO] Using RealSense internal color intrinsics:")
            print(self.intrinsic_matrix)
            print(f"[INFO] Distortion coeffs: {self.dist_coeffs}")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch RealSense intrinsics: {e}")
    
    def _find_chessboard_corners(self, show_corners=False):
        """Find chessboard corners in images"""
        print("[INFO] Finding chessboard corners...")
        chessboard_corners = []
        valid_indices = []
        
        for i, image in enumerate(self.images):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size)
            
            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                chessboard_corners.append(corners)
                valid_indices.append(i)
                
                # Draw and save corners
                corner_image = image.copy()
                cv2.drawChessboardCorners(corner_image, self.pattern_size, corners, ret)
                
                if show_corners:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(corner_image)
                    plt.title(f"Detected corners in image {i}")
                    plt.axis('off')
                    plt.show()
                
                # Save corner detection result
                corner_filename = os.path.join(CORNERS_DIR, f"corners_{i:03d}.png")
                cv2.imwrite(corner_filename, cv2.cvtColor(corner_image, cv2.COLOR_RGB2BGR))
                
                print(f"[INFO] Found corners in image {i}")
            else:
                print(f"[WARN] No chessboard found in image {i}")
        
        print(f"[INFO] Found corners in {len(chessboard_corners)}/{len(self.images)} images")
        return chessboard_corners, valid_indices
    
    def _compute_camera_poses(self):
        """Compute camera poses relative to chessboard target (using internal intrinsics)"""
        print("[INFO] Computing camera poses...")
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2) * self.square_size
        R_target2cam = []
        t_target2cam = []
        for corners in self.chessboard_corners:
            ok, rvec, tvec = cv2.solvePnP(objp, corners, self.intrinsic_matrix, self.dist_coeffs)
            if not ok:
                continue
            Rm, _ = cv2.Rodrigues(rvec)
            R_target2cam.append(Rm)
            t_target2cam.append(tvec)
        return R_target2cam, t_target2cam
    
    def _prepare_hand_eye_data(self):
        """Prepare data for hand-eye calibration"""
        print("[INFO] Preparing hand-eye calibration data...")
        
        # Convert target2cam to cam2target
        self.R_cam2target = [R.T for R in self.R_target2cam]
        self.t_cam2target = [-R.T @ t for R, t in zip(self.R_target2cam, self.t_target2cam)]
        
        # Convert base2ee to ee2base  
        self.R_ee2base = [T[:3, :3].T for T in self.T_base2ee_list]
        self.t_ee2base = [-T[:3, :3].T @ T[:3, 3] for T in self.T_base2ee_list]

        
        # Convert to rodrigues vectors for OpenCV
        self.rvec_ee2base = [cv2.Rodrigues(R)[0] for R in self.R_ee2base]

        self.R_base2ee = [T[:3, :3] for T in self.T_base2ee_list]
        self.t_base2ee = [T[:3, 3].reshape(3,1) for T in self.T_base2ee_list]
        self.rvec_base2ee = [cv2.Rodrigues(R)[0] for R in self.R_base2ee]
    
    def _solve_hand_eye_calibration(self):
        """Solve hand-eye calibration using multiple methods"""
        print("[INFO] Performing hand-eye calibration...")
        
        results = {}
        
        # Test all calibrateHandEye methods
        for method_name, method_flag in METHOD_MAP.items():
            try:
                print(f"\n[INFO] Testing method: {method_name}")
                
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                     
                    self.rvec_base2ee, # Here base2ee means ee2base
                    self.t_base2ee,
                    self.R_target2cam,
                    self.t_target2cam,
                    method=method_flag
                )
                
                # Create 4x4 transformation matrix
                T_cam2gripper = np.concatenate((R_cam2gripper, t_cam2gripper), axis=1)
                T_cam2gripper = np.concatenate((T_cam2gripper, np.array([[0, 0, 0, 1]])), axis=0)
                # T_cam2gripper = np.linalg.inv(T_cam2gripper)  # Invert to get gripper2cam
                # R_cam2gripper = T_cam2gripper[:3, :3]
                # t_cam2gripper = T_cam2gripper[:3, 3].reshape(3,1)

                # Calculate validation error
                error = self._validate_calibration(R_cam2gripper, t_cam2gripper)
                
                results[method_name] = {
                    'R': R_cam2gripper,
                    't': t_cam2gripper,
                    'T': T_cam2gripper,
                    'error': error
                }
                
                print(f"[INFO] {method_name} - Validation error: {error:.6f}")
                print(f"[INFO] Translation: {t_cam2gripper.flatten()}")
                
                # Save results
                np.savez(os.path.join(RESULTS_DIR, f"T_cam2gripper_{method_name}.npz"), T_cam2gripper)
                np.savez(os.path.join(RESULTS_DIR, f"T_gripper2cam_{method_name}.npz"), np.linalg.inv(T_cam2gripper))
                
            except Exception as e:
                print(f"[ERROR] {method_name} failed: {e}")
        
        # Find best result
        if results:
            best_method = min(results.keys(), key=lambda k: results[k]['error'])
            print(f"\n[SUCCESS] Best method: {best_method}")
            print(f"[SUCCESS] Best error: {results[best_method]['error']:.6f}")
            
            # Save best result with special name
            best_T = results[best_method]['T']
            np.savez(os.path.join(RESULTS_DIR, "T_cam2gripper_BEST.npz"), best_T)
            np.savez(os.path.join(RESULTS_DIR, "T_gripper2cam_BEST.npz"), np.linalg.inv(best_T))
            
            # Save summary
            self._save_calibration_summary(results, best_method)
        
        return results
    
    def _validate_calibration(self, R_cam2gripper, t_cam2gripper):
        """Validate hand-eye calibration result"""
        total_error = 0
        num_pairs = len(self.R_cam2target)
        
        for i in range(num_pairs):
            # Transform from camera to base through gripper
            R_cam2base_via_gripper = self.R_ee2base[i].T @ R_cam2gripper
            t_cam2base_via_gripper = self.R_ee2base[i].T @ t_cam2gripper + self.t_ee2base[i]
            
            # Transform from camera to base through target
            R_cam2base_via_target = self.R_cam2target[i]
            t_cam2base_via_target = self.t_cam2target[i]
            
            # Calculate error
            R_error = R_cam2base_via_gripper @ R_cam2base_via_target.T
            angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
            trans_error = np.linalg.norm(t_cam2base_via_gripper - t_cam2base_via_target)
            
            total_error += angle_error + trans_error
        
        return total_error / num_pairs
    
    def _save_calibration_summary(self, results, best_method):
        """Save calibration summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(RESULTS_DIR, f"calibration_summary_{timestamp}.txt")
        
        with open(summary_file, 'w') as f:
            f.write("Hand-Eye Calibration Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Pattern size: {self.pattern_size}\n")
            f.write(f"Square size: {self.square_size}\n")
            f.write(f"Number of samples: {len(self.valid_indices)}\n\n")
            
            f.write("Results by method:\n")
            f.write("-" * 30 + "\n")
            
            for method, data in results.items():
                f.write(f"\n{method}:\n")
                f.write(f"  Error: {data['error']:.6f}\n")
                f.write(f"  Translation: {data['t'].flatten()}\n")
                if method == best_method:
                    f.write("  *** BEST METHOD ***\n")
        
        print(f"[INFO] Calibration summary saved to {summary_file}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.pipeline:
            self.pipeline.stop()

# =========================
# ARUCO DETECTION FUNCTION
# =========================
def pixel_depth_to_base(u, v, Z_m, intrinsics, T_cam2gripper, T_base2ee):
    """
    Convert pixel (u,v) and depth Z_m to robot/world coordinates using RealSense intrinsics and hand-eye calibration.
    Assumes depth is aligned to the color stream and intrinsics are the color intrinsics.
    """
    P_cam = np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], float(Z_m))).reshape(3,1)
    print(f"[DEBUG] P_cam (m): {P_cam.flatten()}")
    P_cam_h = np.vstack((P_cam, [1.0]))
    print(f"[DEBUG] P_cam_h: {P_cam_h.flatten()}")
    # Reverse T_base2ee (i.e., use its inverse)
    # P_base = np.linalg.inv(T_base2ee) @ T_cam2gripper @ P_cam_h
    P_base = T_base2ee @ T_cam2gripper @ P_cam_h
    return P_base[:3].flatten()

def detect_aruco_and_center(calibrator):
    """
    Detect ArUco markers on the current aligned frameset and compute base-frame coordinates
    of the marker center using RealSense deprojection + hand-eye.
    """
    # Grab a single synchronized frameset and align depth to color
    frames = calibrator.pipeline.wait_for_frames()
    aligned = calibrator.align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        print("[ERROR] Failed to get aligned color/depth frames.")
        return None, None

    # Use color frame for detection
    image = np.asanyarray(color_frame.get_data())  # BGR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ArUco detection (DICT_4X4_100 to match your stream script)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print("[ARUCO] No ArUco marker detected.")
        return None, None

    image_out = image.copy()
    ids = ids.flatten()
    centers_px = []
    for idx, marker_corners in zip(ids, corners):
        pts = marker_corners.reshape(-1, 2)
        center = pts.mean(axis=0)
        u, v = int(round(center[0])), int(round(center[1]))
        centers_px.append(center)
        cv2.circle(image_out, (u, v), 6, (255, 0, 0), -1)
        print(f"[ARUCO] Marker ID: {idx}, center pixel: ({center[0]:.1f}, {center[1]:.1f})")

        # Robust depth: median of a 5x5 window (valid > 0)
        h, w = gray.shape[:2]
        x0, x1 = max(0, u-2), min(w-1, u+2)
        y0, y1 = max(0, v-2), min(h-1, v+2)
        zs = []
        for yy in range(y0, y1+1):
            for xx in range(x0, x1+1):
                d = depth_frame.get_distance(xx, yy)
                if d > 0:
                    zs.append(d)
        if not zs:
            print("[ARUCO] Invalid depth neighborhood.")
            continue
        Z_m = float(np.median(zs))
        # Z_m = 0.2265
        print(f"[ARUCO] Depth (median 5x5) at center: {Z_m:.4f} m")

        try:
            # Load best hand-eye
            T_cam2gripper = np.load(os.path.join(RESULTS_DIR, "T_cam2gripper_HORAUD.npz"))["arr_0"]
            # Current robot pose
            tcp_info = calibrator.robot.get_tcp_info(calibrator.rc)
            T_base2ee = get_robot_transform_matrix(tcp_info)
            # Use calibrated intrinsics instead of RealSense intrinsics
            # Convert RealSense intrinsics object to our format for rs2_deproject_pixel_to_point
            rs_intrinsics = calibrator.color_intrinsics
            rs_intrinsics.fx = float(calibrator.intrinsic_matrix[0, 0])
            rs_intrinsics.fy = float(calibrator.intrinsic_matrix[1, 1])
            rs_intrinsics.ppx = float(calibrator.intrinsic_matrix[0, 2])
            rs_intrinsics.ppy = float(calibrator.intrinsic_matrix[1, 2])
            # Update distortion coefficients if available
            if hasattr(calibrator, 'dist_coeffs') and calibrator.dist_coeffs is not None:
                for i in range(min(5, len(calibrator.dist_coeffs.flatten()))):
                    rs_intrinsics.coeffs[i] = float(calibrator.dist_coeffs.flatten()[i])
            
            # Compute base coordinates using updated intrinsics
            P_base = pixel_depth_to_base(u, v, Z_m, rs_intrinsics, T_cam2gripper, T_base2ee)
            print(f"\033[92m[ARUCO] Marker center (base frame) [m]: {P_base}\033[0m")
            print(f"\033[93mError: {P_base[0]*1000-475.8} mm in X, {P_base[1]*1000+18.6} mm in Y, {P_base[2]*1000-2.3} mm in Z (from expected [475.8, -18.6, 2.3] mm)\033[0m")
            move_opt = input("Press 'm' to move robot to marker center, any other key to skip: ").strip().lower()
            if move_opt == 'm':
                move_linear(calibrator.robot, calibrator.rc, [P_base[0]*1000, P_base[1]*1000, 2.5, 90, 0, -90])
        except Exception as e:
            print(f"[WARN] Base transform failed: {e}")

    cv2.aruco.drawDetectedMarkers(image_out, corners, ids.reshape(-1,1))
    cv2.imshow('Aruco Detection (DICT_4X4_100)', image_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ids, centers_px

# =========================
# MAIN FUNCTION
# =========================
def main():
    print("[INFO] Robot Hand-Eye Calibration Tool")
    print("=" * 50)
    
    # Ask for camera config file
    # config_file = input("Enter camera config JSON file path (or press Enter for default): ").strip()
    config_file = "/home/msis/Desktop/Hieu/cam_high_config.json"
    if not config_file:
        config_file = None
    
    collect_new = input("Collect new calibration data? [Y/n]: ").strip().lower()
    collect_new_data = collect_new != 'n'
    show_corners = input("Show detected corners? [Y/n]: ").strip().lower()
    show_corners = show_corners != 'n'
    show_error = input("Show reprojection error plot? [Y/n]: ").strip().lower()
    show_project_error = show_error != 'n'
    try:
        calibrator = RobotHandEyeCalibration(
            collect_new_data=collect_new_data,
            show_corners=show_corners,
            show_project_error=show_project_error,
            camera_config_file=config_file
        )
        print("\n[SUCCESS] Hand-eye calibration completed!")
        aruco_opt = input("Press 'd' to detect ArUco marker on current frame, any other key to exit: ").strip().lower()
        if aruco_opt == 'd':
            detect_aruco_and_center(calibrator)
        else:
            print("[INFO] Exiting without ArUco detection.")
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}")
    finally:
        if 'calibrator' in locals():
            calibrator.cleanup()

if __name__ == "__main__":
    main()
