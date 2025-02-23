import pybullet as p
import numpy as np
from numpy.linalg import norm
from scipy.stats import qmc
import time


def angular_distance(a, b):
    """
    Calculate minimal angular difference in radians
    """
    return np.pi - abs(abs(a - b) - np.pi)


def generateIKDataset(robot, numSamples):
    """
    Generates dataset to use to train and test the ml models

    Args:
        - robot (RobotController): pybullet model with forward kinematics capability
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
    
    # Data collection
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
    return processedPoses, np.array(validJointAngles)

def processEndEffectorPoses(endEffectorPoses):
    """
    Processes the end effector poses
    """
    return np.array([
        [x, y, z, 
         np.sin(roll), np.cos(roll),
         np.sin(pitch), np.cos(pitch),
         np.sin(yaw), np.cos(yaw)]
        for x, y, z, roll, pitch, yaw in endEffectorPoses
    ])


def calculatePoseErrors(yPred, XTest, robot):
    """
    Calculate position and orientation errors between predicted and target poses.
    
    Args:
        - yPred (np.ndarray): Predicted joint angles
        - XTest (np.ndarray): Target poses
        - robot (RobotController): pybullet model with forward kinematics capability
        
    Returns:
        - poseErrors (np.ndarray): Array of shape (n_samples, 2) containing [position_error, orientation_error]
    """
    # Input validation
    assert len(yPred) == len(XTest), "Predictions and targets must have same length"
    assert XTest.shape[1] == 9, "Target poses must have 9 features"
    
    poseErrors = np.empty((len(yPred), 2))
    
    for i, (anglesPred, targetPose) in enumerate(zip(yPred, XTest)):
        # Compute achieved pose
        achievedPose = robot.solveForwardPositonKinematics(anglesPred)
        
        # Handle invalid configurations
        if achievedPose is None:  # Add validity check in your robot class
            poseErrors[i] = [np.nan, np.nan]
            continue
            
        # Position error (Euclidean distance)
        positionError = norm(achievedPose[:3] - targetPose[:3])
        
        # Orientation error calculation
        # Extract true angles from processed features
        targetRoll = np.arctan2(targetPose[3], targetPose[4])
        targetPitch = np.arctan2(targetPose[5], targetPose[6])
        targetYaw = np.arctan2(targetPose[7], targetPose[8])
        
        # Calculate angular errors
        rollError = angular_distance(achievedPose[3], targetRoll)
        pitchError = angular_distance(achievedPose[4], targetPitch)
        yawError = angular_distance(achievedPose[5], targetYaw)
        
        # Combined orientation error (use RMS instead of L2 norm)
        orientationError = np.sqrt(np.mean([rollError**2, 
                                           pitchError**2, 
                                           yawError**2]))
        
        poseErrors[i] = [positionError, orientationError]
    
    return poseErrors


def trainModel(XTrain, yTrain, model):
    """
    Train the ML model

    Args:
        - XTrain (np.array): Training input set
        - yTrain (np.array): Training output set
        - model: ML model
    """
    startTrain = time.time()
    model.fit(XTrain, yTrain)
    endTrain = time.time()
    trainingTime = endTrain - startTrain

    return model, trainingTime


def testModel(XTest, model, scaler):
    """
    Test the ML model

    Args:
        - XTest (np.array): Testing input set
        - model: ML model
        - scaler (StandardScaler): Scales the predicted values to match the scale of the actual values
    """
    startTest = time.time()
    yPred = scaler.inverse_transform(model.predict(XTest))
    endTest = time.time()
    testingTime = endTest - startTest

    return yPred, testingTime
