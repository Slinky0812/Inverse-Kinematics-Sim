# Linear Regression model
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import loguniform

from generate.generate_data import testModel, calculatePoseErrors

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
    """
    # Create pipeline
    lrPipe = make_pipeline(
        scaler,
        Ridge(alpha=1.0, random_state=42)
    )
    
    # Define parameter grid
    paramGrid = {
        'ridge__alpha': np.logspace(-3, 3, 7)
    }

    # Perform grid search
    gridSearch = GridSearchCV(
        lrPipe, 
        paramGrid, 
        cv=3, 
        n_jobs=2,
        scoring='neg_mean_squared_error'
    )
    gridSearch.fit(XTrain, yTrain)
    
    # Find the best model
    bestLR = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Test the best model
    yPred, testingTime = testModel(XTest, bestLR, scaler)

    # Calculate metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, yTest, robot)
    
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
    """
    # Create pipeline
    pipeline = make_pipeline(
        scaler,
        MultiOutputRegressor(
            BayesianRidge(
                compute_score=True,
                verbose=True  # Show convergence progress
            )
        )
    )

    # Create parameter grid
    paramGrid = {
        'multioutputregressor__estimator__alpha_1': loguniform(1e-7, 1e-3),
        'multioutputregressor__estimator__alpha_2': loguniform(1e-7, 1e-3),
        'multioutputregressor__estimator__lambda_1': loguniform(1e-7, 1e-3),
        'multioutputregressor__estimator__lambda_2': loguniform(1e-7, 1e-3),
    }

    # Perform randomized search 
    search = RandomizedSearchCV(
        pipeline,
        paramGrid,
        # n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=2,
        # random_state=42
    )
    search.fit(XTrain, yTrain)

    # Find the best model
    bestBR = search.best_estimator_
    trainingTime = search.cv_results_['mean_fit_time'][search.best_index_]

    # Test the best model
    yPred, testingTime = testModel(XTest, bestBR, scaler)

    # Calculate metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    # Calculate pose errors
    pose_errors = calculatePoseErrors(yPred, yTest, robot)
    
    # Return results
    return pose_errors, mse, mae, trainingTime, testingTime, r2, search.best_params_