# GA-DRL algorithm with robotic manipulator Aubo i5

# GA-DRL paper citation
```
@inproceedings{sehgal2019deep,
  title={Deep reinforcement learning using genetic algorithm for parameter optimization},
  author={Sehgal, Adarsh and La, Hung and Louis, Sushil and Nguyen, Hai},
  booktitle={2019 Third IEEE International Conference on Robotic Computing (IRC)},
  pages={596--601},
  year={2019},
  organization={IEEE}
}
```

## Prerequisite
- Must have compiled the aubo robot github repo under the kinetic branch,which can be found here:
  - It is safe to remove auto_controller folder if you get build error with this package
```
https://github.com/adarshsehgal/aubo_robot
```
- Ubuntu 16.04
- Ros Kinetic
- Python 2.7
- Python verison >= 3.5 (this code was run on python 3.5)
- Aubo gym environment uses python2.7 with moveit
- Genetic Algorithm ga.py uses python3.7
- pip install gym==0.15.6
- Install the packages needed to install gym
```
pip3 install scipy tqdm joblib cloudpickle click opencv-python
```
- pip install tensorflow==1.14.0
- openai_ros
  - IMPORTANT: run rosdep install openai_ros EACH time you run the code (for each terminal)
```
https://github.com/adarshsehgal/openai_ros
```
- update pip3 to 21.0 or latest (if no errors)
```
pip3 install --upgrade pip
```
- To avoid libmoveit_robot_trajectory.so error, follow below commands
  - replace the version number of 0.9.17 with what you have in below directory
  - don’t change 0.9.15 
```
cd /opt/ros/kinetic/lib 
sudo cp -r libmoveit_robot_trajectory.so.0.9.17 .. 
sudo cp -r libmoveit_robot_state.so.0.9.17 ..
cd .. 
sudo mv libmoveit_robot_state.so.0.9.17 libmoveit_robot_state.so.0.9.15 
sudo mv libmoveit_robot_model.so.0.9.17 libmoveit_robot_model.so.0.9.15
sudo mv libmoveit_robot_trajectory.so.0.9.17 libmoveit_robot_trajectory.so.0.9.15
sudo cp -r libmoveit_robot_state.so.0.9.15 lib/ 
sudo cp -r libmoveit_robot_model.so.0.9.15 lib/ 
sudo cp -r libmoveit_robot_trajectory.so.0.9.15 lib/
```
- To avoid error with installation of mpi4py package
```
sudo apt install libpython3.7-dev
pip3 install mpi4py
```
- No need to install mujoco-py, since the code uses Rviz
- Genetic algorithm library
```
pip3 install https://github.com/chovanecm/python-genetic-algorithm/archive/master.zip#egg=mchgenalg
https://github.com/adarshsehgal/python-genetic-algorithm.git
```


## How to run the program


**Before running roslaunch command, setup aubo robot repository with ros kinetic (link in pre requite section)**
```
cd catkin_workspace
catkin build
source devel/setup.bash
rosdep install openai_ros
```
**Training**

First clone the repository
```
git clone <github url> 
```
Training without simulation first, allows for the use of multiple CPU cores. 
To begin training using default values from openai: 
```
cd ~/simulation_branch

python3 train.py
```
The best policy will be generated in the directory `/tmp/newlog`.

To begin training using the GA_her+ddpg parameters generated from the python script `ga.py`
```
python3 train.py --polyak_value=0.924 --gamma_value=0.949 --q_learning=0.001 --pi_learning=0.001 --random_epsilon=0.584 --noise_epsilon=0.232
```

If you would like to recieve your own GA optimized parameters run:
```
python3 ga.py
```
The best parameters will be saved to the file `bestParameters.txt`



**Launch rviz and moveit by either launch commands:** 

**For testing with simulation:**



```
cd ~/catkin_ws 
source devel/setup.bash
roslaunch aubo_i5_moveit_config demo.launch
```
For real robot:
```
roslaunch aubo_i5_robot_config moveit_planning_execution.launch robot_ip:=<your robot ip>
```
Go into the simulation branch directory
```
cd 
cd simulation_branch
```

To run the connection between the robot gym environment and moveit run:
```
python2.7 moveit_motion_control.py
```

Run the genetic algorithm on her+ddpg while still in newher directory:

Gym Environments available for GA-DRL execution (can be changes in ga.py):
- AuboReach-v0 - executes joint states with moveit
- AuboReach-v1 - only calculates actions but does not execute joint states (increased learning speed)


## How to play the environment with any chosen policy file
You can see the environment in action by providing the policy file:
```
python3 -m play.py <file_path>
python3 -m play.py /tmp/newlog/policy_best.pkl
```
where file_path = /tmp/openaiGA/policy_best.pkl in our case  .

## How to plot results:
For one set of parameter values and for one DRL run, plot results using:
```
python3 plot.py <dir>
python3 plot.py /tmp/newlog
```
where, dir = /tmp/newlog in our case, as mentioned in ga.py file. You can provide any log directory.
- If you are testing one set of parameters, in train.py, comment out the code which stops the system as soon as it reaches threshold success rate. 
```
sys.exit()
```
