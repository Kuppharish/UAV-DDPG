#!/usr/bin/env python

import gym
import rospy
import time
import numpy as np
import math
import tf
import time
from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose
#from hector_uav_msgs.msg import Altimeter
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg
from gazebo_msgs.msg import ContactsState as Pressure
from gazebo_msgs.msg import ModelStates, LinkStates
from apriltags2_ros.msg import AprilTagDetectionArray as aprmsg
from dji_sdk.msg import Gimbal
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection
import pandas as pd

#register the training environment in the gym as an available one
reg = register(
    id='QuadcopterLiveShow-v0',
    entry_point='myquadcopter_env:QuadCopterEnv',
    timestep_limit=600,
    )


class QuadCopterEnv(gym.Env):

    def __init__(self):
        
        # We assume that a ROS node has already been created
        # before initialising the environment
        
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.takeoff_pub = rospy.Publisher('/drone/takeoff', EmptyTopicMsg, queue_size=0)
        self.gimbal_pub= rospy.Publisher('/dji_sdk/gimbal',Gimbal,queue_size=5)
        
        # gets training parameters from param server
        '''
        self.speed_value = rospy.get_param("/speed_value")
        self.desired_pose = Pose()
        self.desired_pose.position.z = rospy.get_param("/desired_pose/z")
        self.desired_pose.position.x = rospy.get_param("/desired_pose/x")
        self.desired_pose.position.y = rospy.get_param("/desired_pose/y")
        self.running_step = rospy.get_param("/running_step")
        self.max_incl = rospy.get_param("/max_incl")
        self.max_altitude = rospy.get_param("/max_altitude")
        self.min_altitude = rospy.get_param("/min_altitude")
        '''
        self.speed_value = 1
        self.desired_pose = Pose()
        self.desired_pose.position.z = 0.7
        self.desired_pose.position.x = 14
        self.desired_pose.position.y = 1
        self.running_step = 0.05
        self.max_incl = 0.7
        self.max_altitude = 20
        self.min_altitude = 0.1
        self.max_x=15
        self.max_y=10
        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        
        self.action_space = 2
        self.observation_space=6
        self.reward_range = (-np.inf, np.inf)

        self._seed()
        self.shapestore=[]
        self.shapestore.append(0)
        self.act=np.zeros(3)
        self.pr=0
        self.pdlen = 0
        self.plotpd = pd.DataFrame({'p_x': [], 'p_y': [], 'v_x': [], 'v_y': [], 'time' : []})
        self.tinit = time.time()

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # Resets the state of the environment and returns an initial observation.
    def _reset(self):
        
        # 1st: resets the simulation to initial values
        self.gazebo.resetSim()

        # 2nd: Unpauses simulation
        self.gazebo.unpauseSim()

        # 3rd: resets the robot to initial conditions
        self.check_topic_publishers_connection()
        #self.init_desired_pose()
        self.takeoff_sequence()
        # 4th: takes an observation of the initial condition of the robot
        data_pose, data_vel, data_pr, data_bs, data_imu = self.take_observation()
        state, reward, done = self.process_data(data_pose, data_vel, data_pr, data_bs, data_imu)
        
        # 5th: pauses simulation
        self.gazebo.pauseSim()
        self.pr = 0
        #self.plotpd = pd.DataFrame({'p_x' : [], 'p_y' : [], 'v_x' : [], 'v_y' : []})
        return state
        
    def _step(self, action):

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        
        # 1st, we decide which velocity command corresponds
        self.gazebo.unpauseSim()
        data_pose_1, data_vel_1, data_pr_1, data_bs_1, data_imu_1 = self.take_observation()
        z_vel=-0.2
        coef=1
        if data_pose_1.position.z<1.2:
            z_vel=0
        vel_cmd = Twist()
        '''
        if action == 0: #FORWARD
            vel_cmd.linear.x = self.speed_value
            vel_cmd.angular.z = 0.0
        elif action == 1: #LEFT
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = self.speed_value
        elif action == 2: #RIGHT
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -self.speed_value
        elif action == 3: #Up
            vel_cmd.linear.z = self.speed_value
            vel_cmd.angular.z = 0.0
        elif action == 4: #Down
            vel_cmd.linear.z = -self.speed_value
            vel_cmd.angular.z = 0.0
        '''
        vel_cmd.linear.x = action[0]*coef
        vel_cmd.linear.y = action[1]*coef
        vel_cmd.linear.z = z_vel
        vel_cmd.angular.x = 0.0
        vel_cmd.angular.y = 0.0
        vel_cmd.angular.z = 0.0
        self.act=action

        # Then we send the command to the robot and let it go
        # for running_step seconds
        self.vel_pub.publish(vel_cmd)
        time.sleep(self.running_step)
        data_pose, data_vel, data_pr, data_bs, data_imu = self.take_observation()
        self.set_gimbal(data_bs,data_pose)
        self.gazebo.pauseSim()

        # finally we get an evaluation based on what happened in the sim
        state,reward,done = self.process_data(data_pose, data_vel, data_pr, data_bs, data_imu)

        # Promote going forwards instead if turning
        '''
        if action == 0:
            reward += 100
        elif action == 1 or action == 2:
            reward -= 50
        elif action == 3:
            reward -= 150
        else:
            reward -= 50
        '''
        return state, reward, done, {}

    def take_observation(self):
        data_pose = None
        while data_pose is None:
            try:
                data_pose = rospy.wait_for_message('/drone/gt_pose', Pose, timeout=5)
            except:
                continue
                #rospy.loginfo("Current drone pose not ready yet, retrying for getting robot pose")

        data_vel = None
        while data_vel is None:
            try:
                data_vel = rospy.wait_for_message('/drone/gt_vel', Twist, timeout=5)
            except:
                continue
                #rospy.loginfo("Current drone vel not ready yet, retrying for getting robot vel")
        data_pr = 0
        '''
        data_pose_t=None
        while data_pose_t is None:
            try:
                data_pose_t = rospy.wait_for_message('/tag_detections', aprmsg, timeout=5)
                if len(data_pose_t.detections)!=0:
                    data_pose= data_pose_t.detections[0].pose.pose.pose
                else:
                    continue
            except:
                rospy.loginfo("waiting for april tag message")
        '''
        data_bs=None
        while data_bs is None:
            try:
                data_bs = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
            except:
                continue
                #rospy.loginfo("Current base imu not ready yet, retrying for getting base values")
        data_imu = None
        '''
        while data_imu is None:
            try:
                data_imu = rospy.wait_for_message('/drone/imu', Imu, timeout=5)
            except:
                #continue
                rospy.loginfo("Current drone imu not ready yet, retrying for getting robot imu")
        '''
        return data_pose, data_vel, data_pr, data_bs, data_imu

    def set_gimbal(self, data_bs,data_pose):
        base_pose = data_bs.pose[2].position
        drone_pose=data_pose.position
        x=drone_pose.x-base_pose.x
        y=drone_pose.y-base_pose.y
        z=drone_pose.z-base_pose.z
        pitch=np.arctan(y/z)*180/math.pi
        yaw=np.arctan(x/z)*180/math.pi
        gimmsg=Gimbal()
        gimmsg.pitch=pitch
        gimmsg.yaw=yaw
        gimmsg.roll=0
        self.gimbal_pub.publish(gimmsg)




    def calculate_dist_between_two_Points(self,p_init,p_end):
        a = np.array((p_init.x ,p_init.y, p_init.z))
        b = np.array((p_end.x ,p_end.y, p_end.z))
        
        dist = np.linalg.norm(a-b)
        
        return dist


    def init_desired_pose(self):
        
        current_init_pose, imu = self.take_observation()

        self.best_dist = self.calculate_dist_between_two_Points(current_init_pose.position, self.desired_pose.position)
    

    def check_topic_publishers_connection(self):
        
        rate = rospy.Rate(10) # 10hz
        while(self.takeoff_pub.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Takeoff yet so we wait and try again")
            rate.sleep();
        rospy.loginfo("Takeoff Publisher Connected")

        while(self.vel_pub.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Cmd_vel yet so we wait and try again")
            rate.sleep();
        rospy.loginfo("Cmd_vel Publisher Connected")
        
    def initial_flight(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0
        vel_cmd.linear.y = 0
        vel_cmd.linear.z = 1
        vel_cmd.angular.x = 0.0
        vel_cmd.angular.y = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

    def reset_cmd_vel_commands(self):
        # We send an empty null Twist
        vel_cmd = Twist()
        vel_cmd.linear.z = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)


    def takeoff_sequence(self, seconds_taking_off=1):
        # Before taking off be sure that cmd_vel value there is is null to avoid drifts
        self.reset_cmd_vel_commands()
        
        takeoff_msg = EmptyTopicMsg()
        rospy.loginfo( "Taking-Off Start")
        self.takeoff_pub.publish(takeoff_msg)
        time.sleep(seconds_taking_off)
        #t=time.time()
        #while(time.time()-t<4):
        #    self.initial_flight()
        rospy.loginfo( "Taking-Off sequence completed")
        

    def improved_distance_reward(self, current_pose):
        current_dist = self.calculate_dist_between_two_Points(current_pose.position, self.desired_pose.position)
        #rospy.loginfo("Calculated Distance = "+str(current_dist))
        
        if current_dist < self.best_dist:
            reward = 100
            self.best_dist = current_dist
        elif current_dist == self.best_dist:
            reward = 0
        else:
            reward = -100
            #print "Made Distance bigger= "+str(self.best_dist)
        
        return reward
        
    def process_data(self, data_pos, data_vel, data_pr, data_bs, data_imu):
        base_pose=data_bs.pose[2].position
        base_vel=data_bs.twist[2].linear
        p_x=data_pos.position.x-base_pose.x
        p_y=data_pos.position.y-base_pose.y
        p_z=data_pos.position.z-base_pose.z
        v_x=data_vel.linear.x-base_vel.x
        v_y=data_vel.linear.y-base_vel.y
        self.plotpd = self.plotpd.append({'p_x' : p_x, 'p_y' : p_y, 'v_x' : v_x, 'v_y' : v_y, 'time' : time.time() - self.tinit},ignore_index=True)
        try:
            self.plotpd.to_csv('./data.csv', sep='\t', encoding='utf-8')
            #print(self.plotpd)
        except:
            print("failed")
        #self.pdlen = self.plotpd.shape[0]
        a_x=self.act[0]
        a_y=self.act[1]
        c=0
        if abs(p_x)<0.75 and abs(p_y)<0.75 and p_z<1.2:
            c=1
        self.pr=self.pr+c
        shaping=-100*np.sqrt(p_x**2+p_y**2)-10*np.sqrt(v_x**2+v_y**2)-np.sqrt(a_x**2+a_y**2)+10*c*(2-abs(a_x)-abs(a_y))
        self.shapestore.append(shaping)
        reward=self.shapestore[-1]-self.shapestore[-2]
        state=np.array([p_x,p_y,p_z,v_x,v_y,c])
        #if c==1:
        #    print("positions: ",p_x,p_y,p_z)
        done = False
        #euler = tf.transformations.euler_from_quaternion([data_imu.orientation.x,data_imu.orientation.y,data_imu.orientation.z,data_imu.orientation.w])
        #roll = euler[0]
        #pitch = euler[1]
        #yaw = euler[2]

        #pitch_bad = not(-self.max_incl < pitch < self.max_incl)
        #roll_bad = not(-self.max_incl < roll < self.max_incl)
        altitude_bad_1 = data_pos.position.z > self.max_altitude
        altitude_bad_2 = data_pos.position.z < self.min_altitude
        x_bad=abs(data_pos.position.x)>self.max_x
        #x_bad=False
        y_bad=abs(data_pos.position.y)>self.max_y

        if altitude_bad_1 or altitude_bad_2 or x_bad or y_bad:
            rospy.loginfo ("(Drone flight status is wrong) >>> ("+str(altitude_bad_2)+","+str(x_bad)+","+str(y_bad)+","+str(self.pr)+")")
            done = True 
        return state,reward,done
