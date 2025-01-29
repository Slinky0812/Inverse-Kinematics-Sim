import pybullet as p
import time
import pybullet_data
from pybullet_controller import RobotController

robot = RobotController()
robot.createWorld(view_world=True)