import pybullet as p
import numpy as np
from numpy.linalg import norm
import time


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
