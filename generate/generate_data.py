import pybullet as p
import numpy as np

def generate_ik_dataset(robot, num_samples):    
    # Get joint limits from URDF
    jointLimits = []
    for joint in robot.controllable_joints:
        info = p.getJointInfo(robot.robot_id, joint)
        lower, upper = info[8], info[9]
        jointLimits.append((lower, upper))
    jointLimits = np.array(jointLimits)
    
    # Generate data
    endEffectorPoses = []  # X (input)
    jointAngles = []  # y (output)
    
    for n in range(0, num_samples):
        # Sample random joint angles within limits
        angles = np.random.uniform(
            low=jointLimits[:, 0], 
            high=jointLimits[:, 1]
        )
                
        # Set joints without simulation
        for i, joint in enumerate(robot.controllable_joints):
            p.resetJointState(robot.robot_id, joint, angles[i])
        
        # Get end-effector pose (x, y, z, roll, pitch, yaw)
        eePose = robot.solveForwardPositonKinematics(angles)
        endEffectorPoses.append(eePose)
        jointAngles.append(angles)
    
    return np.array(endEffectorPoses), np.array(jointAngles)