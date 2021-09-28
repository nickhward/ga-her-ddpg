# GA-DRL algorithm with robotic manipulator Aubo i5

##Prerequisite
- Must have compiled the aubo robot github repo under the kinetic branch,which can be found here: 
```
https://github.com/adarshsehgal/aubo_robot
```
- Ubuntu 16.04
- Ros Kinetic
- Python 2.7
- Python verison >= 3.5 (this code was run on python 3.5)
- pip install gym==0.15.6
- pip install tensorflow==1.14.0
- openai_ros
  - IMPORTANT: run rosdep install openai_ros EACH time you run the code (for each terminal)
```
https://github.com/adarshsehgal/openai_ros
```

##How to run the program


**Before running roslaunch command, setup aubo robot repository with ros kinetic (link in pre requite section)**
```
cd catkin_workspace
catkin build
source deve;/setup.bash
rosdep install openai_ros
```

**Launch rviz and moveit by either launch commands:** 

**For simulation:**
```
roslaunch aubo_i5_moveit_config demo.launch
```
For real robot:
```
roslaunch aubo_i5_robot_config moveit_planning_execution.launch robot_ip:=<your robot ip>
```
Go into the newher directory - `cd newher`

To run the connection between the robot gym environment and moveit run:
```
python2.7 moveit_motion_control.py
```

Run the genetic algorithm on her+ddpg while still in newher directory:
```
python3 ga.py
```
