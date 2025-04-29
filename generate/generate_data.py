import pybullet as p
import numpy as np
import time


def encodeAngles(jointAngles):
    """
    Encodes joint angles into sine and cosine values.

    Args:
        - jointAngles (np.ndarray): Joint angles of shape (numSamples, numJoints)
    
    Returns:
        - np.hstack: Encoded angles of shape (numSamples, 2 * numJoints)
    """
    sin_vals = np.sin(jointAngles)
    cos_vals = np.cos(jointAngles)

    return np.hstack((sin_vals, cos_vals))  # Concatenate along feature axis


def decodeAngles(sin_vals, cos_vals):
    """
    Decodes sine and cosine values back into joint angles.
    
    Args:
        - sin_vals (np.ndarray): Sine values of shape (numSamples, numJoints)
        - cos_vals (np.ndarray): Cosine values of shape (numSamples, numJoints)

    Returns:
        - np.arctan2: Decoded angles of shape (numSamples, numJoints)
    """
    return np.arctan2(sin_vals, cos_vals)


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
        positionError = np.linalg.norm(actualPos - predPos)
        
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
    
    if scaler is not None:
        yPred = scaler.inverse_transform(model.predict(XTest))
    else:
        yPred = model.predict(XTest)
    
    # Decode angles to ensure equal weighting in distance calculations
    yPredDecoded = decodeAngles(yPred[:, :7], yPred[:, 7:])
    endTest = time.time()
    testingTime = endTest - startTest

    return yPredDecoded, testingTime
