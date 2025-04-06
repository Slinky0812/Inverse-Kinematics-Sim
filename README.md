# Inverse-Kinematics-Sim
This project will use different machine learning algorithms and apply them to solving the inverse kinematics problem in robotics, and evaluate how effective each of these algorithms are using metrics like their Mean Absolute Error, Mean Squared Error, R² Score, and Pose Error values.

## How to run
Before you can run this program, please ensure you have Microsoft Visual Studio Build Tools installed. 

All the dependencies are found in the requirements.txt, which can be installed via the command line:

`pip install -r requirements.txt`

To run this project, the command is:

`python main.py`

## Credits
The `pybullet_kinematics_dynamics_control` folder is a submodule from the following GitHub repository: https://github.com/deepakraina99/pybullet-kinematics-dynamics-control

The `PandaRobot.jl` folder (found inside of the `urdf` folder) is a submodule from the following GitHub repository: https://github.com/StanfordASL/PandaRobot.jl.git