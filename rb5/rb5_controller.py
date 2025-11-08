def move_to(pose):
    print(f"[RB5] Moving to: {pose}")

def control_gripper(state):
    print(f"[RB5] Gripper: {state}")

def execute(traj):
    move_to(traj["pick"]["position"])
    control_gripper("close")
    move_to(traj["place"]["position"])
    control_gripper("open")
