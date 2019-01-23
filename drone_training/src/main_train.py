#!/usr/bin/env python

import gym
from drone_train import trainer
# ROS packages required
import rospy
import rospkg
import os

# import our training environment
import myquadcopter_env

if __name__ == '__main__':
    
    rospy.init_node('drone_gym', anonymous=True)

    # Create the Gym environment
    env = gym.make('QuadcopterLiveShow-v0')
    rospy.loginfo ( "Gym environment done")
        
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drone_training')
    outdir = pkg_path + '/training_results'
    #env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")
    trainer(env,outdir)
    print("done!!!!!!!!!")
    os.system("rosclean purge")
    os.sysetm("y")

