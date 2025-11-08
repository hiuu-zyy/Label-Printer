from simulation import pickplace
import time

class Arm:
    def __init__(self, env, gui=True, render=False, high_res=False, high_frame_rate=False):
        self.sim = env

    def reset(self):
        self.sim.reset()

    def step(self, steps=1):
        for _ in range(steps):
            self.sim.step_sim(1)

    def control_gripper(self, position):
        self.sim.control_gripper(position)
        for _ in range(240):  # Đợi robot di chuyển & render
            self.sim.step_sim_and_render()
        time.sleep(1)

    def _move_L(self, position):
        self.sim.movep(position)
        for _ in range(240):  # Đợi robot di chuyển & render
            self.sim.step_sim_and_render()
        time.sleep(1)

    def move_down(self, distance=0.1):
        current_pos = self.sim.get_ee_pos()
        new_pos = [current_pos[0], current_pos[1], current_pos[2] - distance]
        self._move_L(new_pos)
        for _ in range(240):  # Đợi robot di chuyển & render
            self.sim.step_sim_and_render()
        time.sleep(1)

    def move_up(self, distance=0.1):
        current_pos = self.sim.get_ee_pos()
        new_pos = [current_pos[0], current_pos[1], current_pos[2] + distance]
        self._move_L(new_pos)
        for _ in range(240):  # Đợi robot di chuyển & render
            self.sim.step_sim_and_render()
        time.sleep(1)

class Gripper:
    def __init__(self, env, render=False, high_res=False, high_frame_rate=False):
        self.sim = env

    def reset(self):
        self.sim.reset()

    def step(self, steps=1):
        for _ in range(steps):
            self.sim.step_sim(1)

    def go(self, pos):
        pos = pos / 100 * 0.025
        self.sim.control_gripper(pos)
        self.sim.try_grasp()
        for _ in range(240):  # Đợi robot di chuyển & render
            self.sim.step_sim_and_render()
        time.sleep(0.5)


