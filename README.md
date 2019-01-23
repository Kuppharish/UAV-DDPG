# UAV_DDPG
cd UAV_DDPG
catkin_make
roslaunch uav_sim simple.launch
rosrun gimbal gazebo_bridge
rosrun drone_training main_train.py