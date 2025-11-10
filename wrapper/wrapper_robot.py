import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from rb5.cobot import *
# from rb5 import cobot
import numpy as np
import rbpodo as rb
import time
from rb5.cobot import *


class SimpleRobot:
    def __init__(self, ip="192.168.0.200"):
        self.ip = ip
        # self.connected = ToCB(self.ip)
        # if not self.connected:
        #     raise Exception("Could not connect to robot.")
        # print("Connected to robot.")
        """Initialize robot connection"""
        try:
            print(f"[INFO] Connecting to robot at {self.ip}...")
            self.robot = ToCB(ip)
            # self.rc = rb.ResponseCollector()
            print("[INFO] Robot connected successfully")
        except Exception as e:
            print(f"[ERROR] Robot connection failed: {e}")
            raise

    def move_linear(self,pose):
        if len(pose) != 6:
            raise Exception("POSE must have exactly 6 elements.")
        
        # added incase of any change in x, y, z, rx, ry, rz
        x = pose[0] 
        y = pose[1]
        z = pose[2]
        rx = pose[3]
        ry = pose[4]
        rz = pose[5]

        if z < 0:
            z = 0

        pose_new = [x,y,z,rx,ry,rz]

        MoveL(float(x), float(y), float(z), float(rx), float(ry), float(rz), float(1200), float(1300))

        while(True):
            if(self.check_if_arrived_at_dest_tcp(pose_new)):
                break

    def check_if_arrived_at_dest_tcp(self,pose):
        # getting current joint position to compare with the target joint position
        # cur_pose = [GetCurrentTCP().x,GetCurrentTCP().y,GetCurrentTCP().z,GetCurrentTCP().rx,GetCurrentTCP().ry,GetCurrentTCP().rz]

        current_tcp = self.getcurrent_TCP()
        if current_tcp is None:
            print("[WARNING] Cannot check arrival - current TCP position is None")
            return False
            
        pose_array_cur = np.array(current_tcp)
        pose_array_given = np.array(pose)

        # Compare the current TCP position with the target TCP position
        return np.allclose(pose_array_cur,pose_array_given, atol=0.05)

    def getcurrent_TCP(self, x=0, y=0, z=0, rx=0, ry=0, rz=0):
        """
        Get current TCP position with error handling.
        
        Returns:
            numpy array of [x, y, z, rx, ry, rz] or None if failed
        """
        try:
            p = GetCurrentTCP()
            if p is None:
                print("[WARNING] GetCurrentTCP() returned None")
                return None
            
            current_position = np.array([
                round(p.x, 2),
                round(p.y, 2),
                round(p.z, 2),
                round(p.rx, 2),
                round(p.ry, 2),
                round(p.rz, 2)
            ], dtype=np.float32)
            return current_position
        except Exception as e:
            print(f"[ERROR] Failed to get current TCP position: {e}")
            return None

    # def move_joint(self, j1, j2, j3, j4, j5, j6, speed=100, acc=500):
    #     MoveJ(j1, j2, j3, j4, j5, j6, speed, acc)
    #     print(f"Moved to joints: {[j1, j2, j3, j4, j5, j6]}")


    def grip(self, pos):
        if not 0 <= int(pos) <= 100:
            raise ValueError("Gripper position must be between 0 and 100.")
        script = f"gripper_macro 6,2,2,0,{str(pos)},0,0,0,0,0"
        ManualScript(script)
        print(f"Gripper moved to {pos}%")

    def initialize_gripper(self, force=80, speed=80, acc=80):
        ManualScript("gripper_macro 6,2,0,0,0,0,0,0,0,0")
        time.sleep(10)
        ManualScript(f"gripper_macro 6,2,1,0,{force},{speed},{acc},0,0,0")
        time.sleep(2)
        print("Gripper initialized")

    def rotate_screw(self):
        ManualScript("digital_out 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1")
        print("Screw rotated")

    def stop_screw_rotation(self):
        ManualScript("digital_out 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1")
        print("Screw rotation stopped")

    def disconnect(self):
        DisConnectToCB()
        print("Disconnected.")

if __name__ == "__main__":
    robot = SimpleRobot("192.168.0.200")
    # robot.initialize_gripper()
    # robot.move_linear([400, 90, 250, 90, 0, 90])
    tcp_info = robot.getcurrent_TCP()
    print("Current TCP:", tcp_info)
    time.sleep(5)
    # robot.grip(50)  # Close halfway
    time.sleep(2)
    # robot.grip(100) # Open fully
    robot.disconnect()
