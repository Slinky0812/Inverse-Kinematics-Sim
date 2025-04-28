# Linear Regression/Bayesian Linear Regression Models
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import loguniform
from sklearn.preprocessing import StandardScaler

from generate.generate_data import testModel, calculatePoseErrors, decodeAngles

import numpy as np


def linearRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a Linear Regression model
    
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
    lrPipe = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    
    # # Define parameter grid
    paramGrid = {
        'linearregression__fit_intercept': [True, False],
        'linearregression__positive': [True, False],
    }

    # Perform grid search
    gridSearch = GridSearchCV(
        lrPipe,
        paramGrid,
        cv=3, 
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    gridSearch.fit(XTrain, yTrain)
    
    # Find the best model
    bestLR = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Test the best model
    yPred, testingTime = testModel(XTest, bestLR, scaler)

    # Inverse transform the actual values to get them back to the original scale
    yTestScaled = scaler.inverse_transform(yTest)
    # Decode angles to ensure equal weighting in distance calculations
    yTestDecode = decodeAngles(yTestScaled[:, :7], yTestScaled[:, 7:])

    # Calculate metrics
    mse = mean_squared_error(yTestDecode, yPred)
    mae = mean_absolute_error(yTestDecode, yPred)
    r2 = r2_score(yTestDecode, yPred)

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, yTestDecode, robot)

    #  VALIDATION - Perform fitting on the training set
    yPredTrain = scaler.inverse_transform(bestLR.predict(XTrain))
    yPredTrainDecode = decodeAngles(yPredTrain[:, :7], yPredTrain[:, 7:])
    minPredTrain = np.min(yPredTrainDecode, axis=0)
    maxPredTrain = np.max(yPredTrainDecode, axis=0)
    print("Training set min:", minPredTrain)
    print("Training set max:", maxPredTrain)

    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2, gridSearch.best_params_


def bayesianLinearRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a Bayesian Linear Regression model
    
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
        - randomSearch.best_params_: Best parameters for the model
    """
    # Create pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(BayesianRidge(compute_score=True, verbose=True))
    )

    # Create parameter grid
    paramGrid = {
        'multioutputregressor__estimator__alpha_1': loguniform(1e-7, 1e-3),
        'multioutputregressor__estimator__alpha_2': loguniform(1e-7, 1e-3),
        'multioutputregressor__estimator__lambda_1': loguniform(1e-7, 1e-3),
        'multioutputregressor__estimator__lambda_2': loguniform(1e-7, 1e-3),
    }

    # Perform randomized search 
    randomSearch = RandomizedSearchCV(
        pipeline,
        paramGrid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
    )
    randomSearch.fit(XTrain, yTrain)

    # Find the best model
    bestBR = randomSearch.best_estimator_
    trainingTime = randomSearch.cv_results_['mean_fit_time'][randomSearch.best_index_]

    # Test the best model
    yPred, testingTime = testModel(XTest, bestBR, scaler)

    # Inverse transform the actual values to get them back to the original scale
    yTestScaled = scaler.inverse_transform(yTest)
    # Decode angles to ensure equal weighting in distance calculations
    yTestDecode = decodeAngles(yTestScaled[:, :7], yTestScaled[:, 7:])

    # Calculate metrics
    mse = mean_squared_error(yTestDecode, yPred)
    mae = mean_absolute_error(yTestDecode, yPred)
    r2 = r2_score(yTestDecode, yPred)

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, yTestDecode, robot)

    # VALIDATION - Perform fitting on the training set
    yPredTrain = scaler.inverse_transform(bestBR.predict(XTrain))
    yPredTrainDecode = decodeAngles(yPredTrain[:, :7], yPredTrain[:, 7:])
    minPredTrain = np.min(yPredTrainDecode, axis=0)
    maxPredTrain = np.max(yPredTrainDecode, axis=0)
    print("Training set min:", minPredTrain)
    print("Training set max:", maxPredTrain)
    
    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2, randomSearch.best_params_