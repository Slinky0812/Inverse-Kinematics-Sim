from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, decodeAngles

import time
import numpy as np


def decisionTree(XTrain, yTrain, XTest, yTest, robot):
    """
    Train and test a Decision Tree model

    Args:
        - XTrain (np.array): Training input set
        - yTrain (np.array): Training output set
        - XTest (np.array): Testing input set
        - yTest (np.array): Testing output set
        - robot (RobotController): Robot object
        - scaler (StandardScalar): Scaler object

    Returns:
        - poseErrors (np.array): Array of position and orientation errors
        - mse (float): Mean Squared Error
        - mae (float): Mean Absolute Error
        - trainingTime (float): Training time
        - testingTime (float): Testing time
        - r2 (float): R² score
        - gridSearch.best_params_: Best parameters for the model
    """
    # Create pipeline
    dtPipe = make_pipeline(
        DecisionTreeRegressor()
    )

    # Define parameter grid
    paramGrid = {
        'decisiontreeregressor__max_depth': [None, 5, 10, 15, 20],
        'decisiontreeregressor__min_samples_split': [2, 5, 10],
        'decisiontreeregressor__min_samples_leaf': [1, 2, 4],
        'decisiontreeregressor__max_features': ['sqrt', 'log2', None],
        'decisiontreeregressor__ccp_alpha': [0.0, 0.01, 0.1]
    }

    # Perform grid search
    gridSearch = GridSearchCV(
        dtPipe,
        paramGrid,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
    )
    gridSearch.fit(XTrain, yTrain)

    # Find the best model
    bestDT = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Test the best model
    startTest = time.time()
    yPred = bestDT.predict(XTest)
    yPred = decodeAngles(yPred[:, :7], yPred[:, 7:])  # Decode angles to match the original scale
    endTest = time.time()
    testingTime = endTest - startTest
    
    # Decode angles to ensure equal weighting in distance calculations
    yTest = decodeAngles(yTest[:, :7], yTest[:, 7:])

    # Calculate metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    # Pose errors
    # poseErrors = calculatePoseErrors(yPred, yTest, robot)
    poseErrors = np.zeros((yPred.shape[0], 6))
    
    minPred = np.min(yPred, axis=0)
    maxPred = np.max(yPred, axis=0)

    # perform fitting for the training set
    yPredTrain = bestDT.predict(XTrain)
    yPredTrainDecode = decodeAngles(yPredTrain[:, :7], yPredTrain[:, 7:])
    minPredTrain = np.min(yPredTrainDecode, axis=0)
    maxPredTrain = np.max(yPredTrainDecode, axis=0)
    print("Training set min:", minPredTrain)
    print("Training set max:", maxPredTrain)

    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2, gridSearch.best_params_, maxPred, minPred