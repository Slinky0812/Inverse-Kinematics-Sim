import pybullet as p
import numpy as np
from numpy.linalg import norm
from scipy.stats import qmc
import time


def generateIKDataset(robot, numSamples):
    """
    Generates dataset to use to train and test the ML models

    Args:
        - robot (RobotController): Robot object
        - numSamples (int): number of data points to be generated

    Returns:
        - processedPoses (np.array): array of arm poses
        - validJointAngles (np.array): array of joint angles
    """
    # Get joint limits
    jointLimits = np.array([p.getJointInfo(robot.robot_id, joint)[8:10] 
                           for joint in robot.controllable_joints])
    
    # Generate Halton samples
    sampler = qmc.Halton(d=len(robot.controllable_joints), scramble=True)
    samples = sampler.random(n=numSamples)
    haltonJointAngles = qmc.scale(samples, jointLimits[:, 0], jointLimits[:, 1])
    
    # Create lists to store valid poses and joint angles
    validPoses = []
    validJointAngles = []
    
    for angles in haltonJointAngles:
        # Set joint states
        for i, joint in enumerate(robot.controllable_joints):
            p.resetJointState(robot.robot_id, joint, angles[i])
        
        # Check singularity
        jacobian = robot.getJacobian(angles)
        if np.linalg.cond(jacobian) > 1e6:  # Proper singularity threshold
            continue
        
        # Compute and store
        eePose = robot.solveForwardPositonKinematics(angles)
        validPoses.append(eePose)
        validJointAngles.append(angles)
    
    # Process data
    processedPoses = processEndEffectorPoses(validPoses)

    # Return data
    return processedPoses, np.array(validJointAngles)


def processEndEffectorPoses(endEffectorPoses):
    """
    Processes the end effector poses

    Args:
        - endEffectorPoses (np.array): array of end effector poses
    """
    # Extract features
    return np.array([
        [x, y, z, 
         np.sin(roll), np.cos(roll),
         np.sin(pitch), np.cos(pitch),
         np.sin(yaw), np.cos(yaw)]
        for x, y, z, roll, pitch, yaw in endEffectorPoses
    ])


def calculatePoseErrors(yPred, yTest, robot):
    """
    Calculate position and orientation errors between predicted and target poses.
    
    Args:
        - yPred (np.ndarray): Predicted joint angles
        - yTest (np.ndarray): Target joint angles
        - robot (RobotController): Robot object
        
    Returns:
        - poseErrors (np.ndarray): Array of shape (numSamples, 2) containing [position_error, orientation_error]
    """
    # Find number of samples
    numSamples = yPred.shape[0]

    # Initialize array to store errors
    poseErrors = np.zeros((numSamples, 2))
    
    for i in range(numSamples):
        # Get actual pose
        robot.setJointPosition(yTest[i])
        actualState = p.getLinkState(robot.robot_id, robot.end_eff_index)
        actualPos = np.array(actualState[0]) # Position
        actualRot = np.array(actualState[1]) # Orientation
        
        # Get predicted pose
        robot.setJointPosition(yPred[i])
        predState = p.getLinkState(robot.robot_id, robot.end_eff_index)
        predPos = np.array(predState[0]) # Position
        predRot = np.array(predState[1]) # Orientation
        
        # Calculate position error (Euclidean distance)
        positionError = norm(actualPos - predPos)
        
        # Calculate orientation error (quaternion angular difference)
        dotProduct = np.clip(np.abs(np.dot(actualRot, predRot)), 0, 1)
        orientationError = 2 * np.arccos(dotProduct)
        
        # Store errors
        poseErrors[i] = [positionError, orientationError]
    
    # Return errors
    return poseErrors


def computePoseErrors(yPred, yTest, robot):
    # Find number of samples
    numSamples = yPred.shape[0]

    # Initialize array to store errors
    poseErrors = np.zeros((numSamples, 2))
    

    for i, (jointAnglesTrue, jointAnglesPred) in enumerate(zip(yTest, yPred)):
        # Get actual pose
        robot.setJointPosition(jointAnglesTrue)
        actualState = p.getLinkState(robot.robot_id, robot.end_eff_index)
        actualPos = np.array(actualState[0]) # Position
        actualRot = np.array(actualState[1]) # Orientation
        
        # Get predicted pose
        robot.setJointPosition(jointAnglesPred)
        predState = p.getLinkState(robot.robot_id, robot.end_eff_index)
        predPos = np.array(predState[0]) # Position
        predRot = np.array(predState[1]) # Orientation
        
        # Position error (Euclidean distance)
        posError = np.linalg.norm(np.array(actualPos) - np.array(predPos))

        # Orientation error (Quaternion distance)
        oriError = 1 - np.abs(np.dot(actualRot, predRot))  # 1 - |q1⋅q2|

        # Store errors
        poseErrors[i] = [posError, oriError]
    
    # Return errors
    return poseErrors


def testModel(XTest, model, scaler):
    """
    Test the ML model

    Args:
        - XTest (np.array): Testing input set
        - model: ML model
        - scaler (StandardScaler): Scales the predicted values to match the scale of the actual values

    Returns:
        - yPred (np.array): Predicted output set
        - testingTime (float): Testing time
    """
    startTest = time.time()
    yPred = scaler.inverse_transform(model.predict(XTest))
    endTest = time.time()
    testingTime = endTest - startTest

    return yPred, testingTime
