import pybullet as p
import time
import pybullet_data

from pybullet_controller import RobotController
from generate.generate_data import generate_ik_dataset

def main():
    robot = RobotController()
    robot.createWorld(view_world=False)

    # generate data set
    # X = the end effector pose
    # y = the joint angles
    X, y = generate_ik_dataset(robot, num_samples=10)

    # for i in range(len(X)):
    #     print(f"X: {X[i]}")
    #     print(f"y: {y[i]}")
    #     print("")

if __name__ == "__main__":
    main()