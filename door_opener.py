import numpy as np
from gym import utils
#import mujoco_env
from gym.envs.mujoco import mujoco_env

from gym.envs.registration import register
import math
#import pyautogui
#import imutils
#import cv2
#import time

register(
    id='DoorOpen-v1',
    entry_point='door_opener:ReacherEnv',
    max_episode_steps=150,
    reward_threshold=-3.75,
)

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'aubo_i5.xml', 2)

    def step(self, a):
        vec = self.get_body_com("right_gripper_link")-self.get_body_com("target")
	#Euclidean distance between gripper and target
        reward_dist = - np.linalg.norm(vec)

	#Reward for the action
        #reward_ctrl = - np.square(a).sum()
        reward_ctrl = - np.transpose(a) * a
        reward_ctrl = - np.square(reward_ctrl).sum()
        #print(reward_ctrl)

	#Reward for door opening
        vec1 = self.get_body_com("right_wheel_Link") - self.get_body_com("target") 
        reward_door = np.linalg.norm(vec1)
        #reward_door=math.pow(reward_door,5)
        #reward_door = reward_door * reward_door
        #print(self.get_body_com("door_surface"))
        #print(reward_door)

	#Total Reward
        w1=0.2
        w2=0.2
        w3=0.6
        reward = w1*reward_dist + w2*reward_ctrl + w3*reward_door

        #t = time.localtime()
        #timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
        #image = pyautogui.screenshot()
        #image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        #resizedimage = cv2.resize(image,(256,256))
        #cv2.imwrite("/home/labuser/DeepLearning/screenshots/"+str(timestamp)+".jpg", resizedimage)
        
        #camid = self.get_cam("gripper_camera")
        #print(self.model.cam_pos[camid])
                
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        #if (abs(reward_dist) < 0.09):
         #   done = True
        #else:
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 13
        self.viewer.cam.elevation = -40
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        #self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        #qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos = self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.1, high=.1, size=2)
            if np.linalg.norm(self.goal) < 0.02:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("right_gripper_link") - self.get_body_com("target")
        ])