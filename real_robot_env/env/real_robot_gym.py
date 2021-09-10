#!/usr/bin/env python

# Python
import copy
import numpy as np
import math
import sys
import time

# ROS 
import rospy



# For moveit connections
from moveit_commander import RobotCommander, MoveGroupCommander, move_group
from moveit_commander import PlanningSceneInterface, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Grasp, GripperTranslation, PlaceLocation
from trajectory_msgs.msg import JointTrajectoryPoint


# ROS msg
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from std_srvs.srv import Empty

# Gym
import gym
from gym import utils, spaces
from gym.utils import seeding
# For register my env
from gym.envs.registration import register

# For inherit RobotGazeboEnv
import real_robot_env_goal




rospy.loginfo("register...")
#register the training environment in the gym as an available one
reg = gym.envs.register(
    id='realRobotEnv-v0',
    entry_point='real_robot_env.env.real_robot_gym:RealRobotRL', # Its directory associated with importing in other sources like from 'ur_reaching.env.ur_sim_env import *' 
    #timestep_limit=100000,
    )

class RealRobotRL(real_robot_env_goal.RealRobotEnv):
    def __init__(self):
        rospy.logdebug("Starting RealRobotRL Class object...")

        # Subscribe joint state and target pose
        rospy.Subscriber("/joint_states", JointState, self.joints_state_callback)

        # Gets training parameters from param server
        self.desired_pose = Pose()
        self.running_step = rospy.get_param("/running_step")
        self.max_height = rospy.get_param("/max_height")
        self.min_height = rospy.get_param("/min_height")
        self.observations = rospy.get_param("/observations")
        
        # Joint limitation
        shp_max = rospy.get_param("/joint_limits_array/shp_max")
        shp_min = rospy.get_param("/joint_limits_array/shp_min")
        shl_max = rospy.get_param("/joint_limits_array/shl_max")
        shl_min = rospy.get_param("/joint_limits_array/shl_min")
        elb_max = rospy.get_param("/joint_limits_array/elb_max")
        elb_min = rospy.get_param("/joint_limits_array/elb_min")
        wr1_max = rospy.get_param("/joint_limits_array/wr1_max")
        wr1_min = rospy.get_param("/joint_limits_array/wr1_min")
        wr2_max = rospy.get_param("/joint_limits_array/wr2_max")
        wr2_min = rospy.get_param("/joint_limits_array/wr2_min")
        wr3_max = rospy.get_param("/joint_limits_array/wr3_max")
        wr3_min = rospy.get_param("/joint_limits_array/wr3_min")
        self.joint_limits = {"shp_max": shp_max,
                             "shp_min": shp_min,
                             "shl_max": shl_max,
                             "shl_min": shl_min,
                             "elb_max": elb_max,
                             "elb_min": elb_min,
                             "wr1_max": wr1_max,
                             "wr1_min": wr1_min,
                             "wr2_max": wr2_max,
                             "wr2_min": wr2_min,
                             "wr3_max": wr3_max,
                             "wr3_min": wr3_min
                             }

        shp_init_value = rospy.get_param("/init_joint_pose/shp")
        shl_init_value = rospy.get_param("/init_joint_pose/shl")
        elb_init_value = rospy.get_param("/init_joint_pose/elb")
        wr1_init_value = rospy.get_param("/init_joint_pose/wr1")
        wr2_init_value = rospy.get_param("/init_joint_pose/wr2")
        wr3_init_value = rospy.get_param("/init_joint_pose/wr3")
        self.init_joint_pose = [shp_init_value, shl_init_value, elb_init_value, wr1_init_value, wr2_init_value, wr3_init_value]

        # Fill in the Done Episode Criteria list
        self.episode_done_criteria = rospy.get_param("/episode_done_criteria")
        

        # We init the observations
        self.base_orientation = Quaternion()
        self.target_point = Point()
        self.joints_state = JointState()
        self.end_effector = Point() 
        self.distance = 0.

        
        
        

        # Gym interface and action
        self.action_space = spaces.Discrete(6)
        self.observation_space = 15 #np.arange(self.get_observations().shape[0])
        self.reward_range = (-np.inf, np.inf)
        self._seed()

        # Change the controller type 

        # Helpful False
        self.stop_flag = False
        stop_trainning_server = rospy.Service('/stop_training', SetBool, self._stop_trainnig)
        start_trainning_server = rospy.Service('/start_training', SetBool, self._start_trainnig)

    def check_stop_flg(self):
        if self.stop_flag is False:
            return False
        else:
            return True

    def _start_trainnig(self, req):
        rospy.logdebug("_start_trainnig!!!!")
        self.stop_flag = False
        return SetBoolResponse(True, "_start_trainnig")

    def _stop_trainnig(self, req):
        rospy.logdebug("_stop_trainnig!!!!")
        self.stop_flag = True
        return SetBoolResponse(True, "_stop_trainnig")
        '''
    def _set_vel_ctrl(self, req):
        rospy.wait_for_service('set_velocity_controller')
        self._ctrl_conn.stop_controllers(self.vel_traj_controller)
        self._ctrl_conn.start_controllers(self.vel_controller)
        self._ctrl_type = 'vel'
        return SetBoolResponse(True, "_set_vel_ctrl")

    def _set_traj_vel_ctrl(self, req):
        rospy.wait_for_service('set_trajectory_velocity_controller')
        self._ctrl_conn.stop_controllers(self.vel_controller)
        self._ctrl_conn.start_controllers(self.vel_traj_controller)    
        self._ctrl_type = 'traj_vel'
        return SetBoolResponse(True, "_set_traj_vel_ctrl")  
        '''

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ''' 
    def link_state_callback(self, msg):
        self.link_state = msg
        self.end_effector = self.link_state.pose[8]
            
    def target_point_callback(self, msg):
        self.target_point = msg
    '''
    def check_all_systems_ready(self):
        """
        We check that all systems are ready
        :return:
        """
        joint_states_msg = None
        while joint_states_msg is None and not rospy.is_shutdown():
            try:
                joint_states_msg = rospy.wait_for_message("/joint_states", JointState, timeout=0.1)
                self.joints_state = joint_states_msg
                rospy.logdebug("Current joint_states READY")
            except Exception as e:
                #self._ctrl_conn.load_controllers("joint_state_controller")
                self._ctrl_conn.start_controllers(controllers_on="joint_state_controller")                
                rospy.logdebug("Current joint_states not ready yet, retrying==>"+str(e))
        '''
        target_pose_msg = None
        while target_pose_msg is None and not rospy.is_shutdown():
            try:
                target_pose_msg = rospy.wait_for_message("/target_blocks_pose", Point, timeout=0.1)
                self.target_point = target_pose_msg
                rospy.logdebug("Reading target pose READY")
            except Exception as e:
                rospy.logdebug("Reading target pose not ready yet, retrying==>"+str(e))
                '''

        rospy.logdebug("ALL SYSTEMS READY")
    '''
    def get_xyz(self, q):
        """Get x,y,z coordinates 
        Args:
            q: a numpy array of joints angle positions.
        Returns:
            xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        return xyz
    
    def get_current_xyz(self):
        """Get x,y,z coordinates according to currrent joint angles
        Returns:
        xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        joint_states = self.joints_state
        shp_joint_ang = joint_states.position[0]
        shl_joint_ang = joint_states.position[1]
        elb_joint_ang = joint_states.position[2]
        wr1_joint_ang = joint_states.position[3]
        wr2_joint_ang = joint_states.position[4]
        wr3_joint_ang = joint_states.position[5]
        
        q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        return xyz
            
    def get_orientation(self, q):
        """Get Euler angles 
        Args:
            q: a numpy array of joints angle positions.
        Returns:
            xyz are the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        orientation = mat[0:3, 0:3]
        roll = -orientation[1, 2]
        pitch = orientation[0, 2]
        yaw = -orientation[0, 1]
        
        return Vector3(roll, pitch, yaw)
    
    def cvt_quat_to_euler(self, quat):
        euler_rpy = Vector3()
        euler = tf.transformations.euler_from_quaternion(
            [self.quat.x, self.quat.y, self.quat.z, self.quat.w])

        euler_rpy.x = euler[0]
        euler_rpy.y = euler[1]
        euler_rpy.z = euler[2]
        return euler_rpy
    '''

    def init_joints_pose(self, init_pos):
        """
        We initialise the Position variable that saves the desired position where we want our
        joints to be
        :param init_pos:
        :return:
        """
        self.current_joint_pose =[]
        self.current_joint_pose = copy.deepcopy(init_pos)
        return self.current_joint_pose

    def get_euclidean_dist(self, p_in, p_pout):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((p_in.x, p_in.y, p_in.z))
        b = np.array((p_pout.x, p_pout.y, p_pout.z))

        distance = np.linalg.norm(a - b)

        return distance

    def joints_state_callback(self,msg):
        self.joints_state = msg

    def get_observations(self):
        """
        Returns the state of the robot needed for OpenAI QLearn Algorithm
        The state will be defined by an array
        :return: observation
        """
        joint_states = self.joints_state
        shp_joint_ang = joint_states.position[0]
        shl_joint_ang = joint_states.position[1]
        elb_joint_ang = joint_states.position[2]
        wr1_joint_ang = joint_states.position[3]
        wr2_joint_ang = joint_states.position[4]
        wr3_joint_ang = joint_states.position[5]
        q = [shp_joint_ang, shl_joint_ang, elb_joint_ang, wr1_joint_ang, wr2_joint_ang, wr3_joint_ang]
        eef_x, eef_y, eef_z = self.get_xyz(q)

        observation = []
        rospy.logdebug("List of Observations==>"+str(self.observations))
        for obs_name in self.observations:
            if obs_name == "shp_joint_ang":
                observation.append(shp_joint_ang)
            elif obs_name == "shl_joint_ang":
                observation.append(shl_joint_ang)
            elif obs_name == "elb_joint_ang":
                observation.append(elb_joint_ang)
            elif obs_name == "wr1_joint_ang":
                observation.append(wr1_joint_ang)
            elif obs_name == "wr2_joint_ang":
                observation.append(wr2_joint_ang)
            elif obs_name == "wr3_joint_ang":
                observation.append(wr3_joint_ang)
            else:
                raise NameError('Observation Asked does not exist=='+str(obs_name))

        return observation
    '''
    def clamp_to_joint_limits(self):
        """
        clamps self.current_joint_pose based on the joint limits
        self._joint_limits
        {
         "shp_max": shp_max,
         "shp_min": shp_min,
         ...
         }
        :return:
        """

        rospy.logdebug("Clamping current_joint_pose>>>" + str(self.current_joint_pose))
        shp_joint_value = self.current_joint_pose[0]
        shl_joint_value = self.current_joint_pose[1]
        elb_joint_value = self.current_joint_pose[2]
        wr1_joint_value = self.current_joint_pose[3]
        wr2_joint_value = self.current_joint_pose[4]
        wr3_joint_value = self.current_joint_pose[5]

        self.current_joint_pose[0] = max(min(shp_joint_value, self._joint_limits["shp_max"]),
                                         self._joint_limits["shp_min"])
        self.current_joint_pose[1] = max(min(shl_joint_value, self._joint_limits["shl_max"]),
                                         self._joint_limits["shl_min"])
        self.current_joint_pose[2] = max(min(elb_joint_value, self._joint_limits["elb_max"]),
                                         self._joint_limits["elb_min"])
        self.current_joint_pose[3] = max(min(wr1_joint_value, self._joint_limits["wr1_max"]),
                                         self._joint_limits["wr1_min"])
        self.current_joint_pose[4] = max(min(wr2_joint_value, self._joint_limits["wr2_max"]),
                                         self._joint_limits["wr2_min"])
        self.current_joint_pose[5] = max(min(wr3_joint_value, self._joint_limits["wr3_max"]),
                                         self._joint_limits["wr3_min"])

        rospy.logdebug("DONE Clamping current_joint_pose>>>" + str(self.current_joint_pose))
    '''
    # Resets the state of the environment and returns an initial observation.
    def reset(self):
        
        #resets the joint poses of the real robot arm to 0. 
        scene = PlanningSceneInterface()
        robot = RobotCommander()
        aubo_arm = MoveGroupCommander('aubo_arm')
        rospy.sleep(1)
        aubo_arm.set_named_target('resting')
        aubo_arm.go()

        grasps = []
        g = Grasp()
        g.id = 'test'
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = 'base_link'
        grasp_pose.pose.position.x = 0
        grasp_pose.pose.position.y = 0
        grasp_pose.pose.position.z = 0
        grasp_pose.pose.orientation.x = 0
        grasp_pose.pose.orientation.y = 0
        grasp_pose.pose.orientation.z = 0
        grasp_pose.pose.orientation.w = 0

        aubo_arm.set_pose_target(grasp_pose)
        aubo_arm.go(wait=True)
        aubo_arm.stop()
        # We save that position as the current joint desired position
        init_pos = self.init_joints_pose(self.init_joint_pose)

        # 4th: We Set the init pose to the jump topic so that the jump control can update
        # We check the jump publisher has connectio
        

        # 5th: Check all subscribers work.
        # Get the state of the Robot defined by its RPY orientation, distance from
        # desired point, contact force and JointState of the three joints
        rospy.logdebug("check_all_systems_ready...")
        self.check_all_systems_ready()

        # 6th: We restore the gravity to original
        
        # self._init_obj_pose()

        # 8th: Get the State Discrete Stringuified version of the observations
        rospy.logdebug("get_observations...")
        observation = self.get_observations()

        return observation

    def _act(self, action):
        scene = PlanningSceneInterface()
        robot = RobotCommander()
        aubo_arm = MoveGroupCommander('aubo_arm')
        rospy.sleep(1)
        aubo_arm.set_named_target('resting')
        

        joint_goal = aubo_arm.get_current_joint_values()
        joint_goal[0] = action[0]
        joint_goal[1] = action[1]
        joint_goal[2] = action[2]
        joint_goal[3] = action[3]
        joint_goal[4] = action[4]
        joint_goal[5] = action[5]
        
        aubo_arm.go(joint_goal, wait=True)
        aubo_arm.stop()


    def training_ok(self):
        rate = rospy.Rate(1)
        while self.check_stop_flg() is True:                  
            rospy.logdebug("stop_flag is ON!!!!") 

            if self.check_stop_flg() is False:
                break 
            rate.sleep()
                
    def step(self, action):
        '''
        ('action: ', array([ 0.,  0. , -0., -0., -0. , 0. ], dtype=float32))        
        '''
        rospy.logdebug("AUBO step func")

        self.training_ok()

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot

        # Act
        self._act(action)
        
        # Then we send the command to the robot and let it go
        # for running_step seconds
        time.sleep(self.running_step)

        # We now process the latest data saved in the class state to calculate
        # the state and the rewards. This way we guarantee that they work
        # with the same exact data.
        # Generate State based on observations
        observation = self.get_observations()

        # finally we get an evaluation based on what happened in the sim
        reward = self.compute_dist_rewards()
        done = self.check_done()

        return observation, reward, done, {}

    def compute_dist_rewards(self):
        #print ("[self.target_point]: ", [self.target_point.x, self.target_point.y, self.target_point.z])
        #print ("(self.get_current_xyz(): ", self.get_current_xyz())

        robot = RobotCommander()
        aubo_arm = MoveGroupCommander('aubo_arm')
        rospy.sleep(1)
        aubo_arm.set_named_target('resting')
        
        aubo_current_pose = aubo_arm.get_current_pose().pose
        

        end_effector_pos = np.array([aubo_current_pose.position.x, aubo_current_pose.position.y, aubo_current_pose.position.z])
        self.distance = np.linalg.norm(end_effector_pos - [0.450977, 0, 0.505381], axis=0)
        return np.exp(-np.linalg.norm(end_effector_pos, axis=0))
        #return np.exp(np.linalg.norm(self.get_current_xyz() - [self.target_point.x, self.target_point.y, self.target_point.z], axis=0))
        m
    def check_done(self):
        if self.distance < 0.1:
            return True
        else :
        	return False
