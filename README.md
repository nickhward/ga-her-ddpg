# ga-drl-aubo

**Prerequisite**
- Must have compiled the aubo robot github repo under the kinetic branch,which can be found here: https://github.com/AuboRobot/aubo_robot
- Ros Kinetic
- Python 2.7
- Python verison >= 3.5 (I personal ran this code on python 3.5)
- pip install gym==0.15.6
- pip install tensorflow==1.14.0


Launch rviz and moveit by either launch commands: 

For simulation:
```
roslaunch aubo_i5_robot_config demo.launch
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
