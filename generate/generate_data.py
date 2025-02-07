import pybullet as p
import numpy as np

def generateIKDataset(robot, num_samples):    
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

    # process end effector poses
    processedEndEffectorPoses = processEndEffectorPoses(endEffectorPoses)
    
    return np.array(processedEndEffectorPoses), np.array(jointAngles)


def processEndEffectorPoses(endEffectorPoses):
    # if 6 values in pose, then it is a single pose
    if len(endEffectorPoses) == 6:
        x, y, z, roll, pitch, yaw = endEffectorPoses
        X_processed = [
            x, y, z,
            np.sin(roll), np.cos(roll),  # Replace roll with sin(roll), cos(roll)
            np.sin(pitch), np.cos(pitch),  # Same for pitch/yaw
            np.sin(yaw), np.cos(yaw)
        ]
    else:
        X_processed = []
        for pose in endEffectorPoses:
            x, y, z, roll, pitch, yaw = pose
            features = [
                x, y, z,
                np.sin(roll), np.cos(roll),  # Replace roll with sin(roll), cos(roll)
                np.sin(pitch), np.cos(pitch),  # Same for pitch/yaw
                np.sin(yaw), np.cos(yaw)
            ]
            X_processed.append(features)
        X_processed = np.array(X_processed)

    return X_processed