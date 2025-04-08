# k-Nearest Neighbors model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from generate.generate_data import calculatePoseErrors, testModel, decodeAngles

import numpy as np


def kNN(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a k-Nearest Neighbours model

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
    knnPipe = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor()
    )
    
    # define parameter grid
    paramGrid = {
        'kneighborsregressor__n_neighbors': list(range(1, 51, 1)),
        'kneighborsregressor__weights': ['uniform', 'distance'],
        'kneighborsregressor__p': [1, 2]
    }
    
    # Perform grid search
    gridSearch = GridSearchCV(
        knnPipe, 
        paramGrid, 
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        refit='MSE', 
    )
    gridSearch.fit(XTrain, yTrain)

    # Find the best model
    bestKNN = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
    
    # Test the best model
    yPred, testingTime = testModel(XTest, bestKNN, scaler)

    # Decode angles to ensure equal weighting in distance calculations
    yTestDecode = decodeAngles(yTest[:, :7], yTest[:, 7:])

    # Calculate metrics
    mse = mean_squared_error(yTestDecode, yPred)
    mae = mean_absolute_error(yTestDecode, yPred)
    r2 = r2_score(yTestDecode, yPred)

    # Pose errors
    # poseErrors = calculatePoseErrors(yPred, yTestDecode, robot)
    poseErrors = np.zeros((yPred.shape[0], 6))

    maxPred = np.max(yPred, axis=0)
    minPred = np.min(yPred, axis=0)

    # perform fitting for the training set
    yPredTrain = scaler.inverse_transform(bestKNN.predict(XTrain))
    yPredTrainDecode = decodeAngles(yPredTrain[:, :7], yPredTrain[:, 7:])
    minPredTrain = np.min(yPredTrainDecode, axis=0)
    maxPredTrain = np.max(yPredTrainDecode, axis=0)
    print("Training set min:", minPredTrain)
    print("Training set max:", maxPredTrain)

    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2, gridSearch.best_params_, maxPred, minPred