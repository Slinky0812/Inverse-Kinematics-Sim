# Linear Regression model
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import loguniform

from generate.generate_data import calculatePoseErrors, trainModel, testModel

import numpy as np


def linearRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Linear Regression with uncertainty quantification and hyperparameter tuning.
    
    Args:
        XTrain, yTrain: Training data
        XTest, yTest: Test data
        robot: Robot model for error calculation
        scaler: Pre-trained scaler
    
    Returns:
        Tuple containing pose errors, metrics, and timings
    """
    # Create scaled pipeline
    lrPipe = make_pipeline(
        scaler,
        Ridge(alpha=1.0, random_state=42)
    )
    
    # Hyperparameter tuning
    paramGrid = {'ridge__alpha': np.logspace(-3, 3, 7)}
    gridSearch = GridSearchCV(lrPipe, paramGrid, cv=5, n_jobs=-1)
    gridSearch.fit(XTrain, yTrain)
    
    # Get best model
    bestLR = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Test model
    yPred, testingTime = testModel(XTest, bestLR, scaler)

    # Metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    
    print(f"Best alpha: {gridSearch.best_params_['ridge__alpha']}")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    poseErrors = calculatePoseErrors(yPred, yTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2


def bayesianLinearRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Bayesian Linear Regression with uncertainty quantification and hyperparameter tuning.
    
    Args:
        XTrain, yTrain: Training data
        XTest, yTest: Test data
        robot: Robot model for error calculation
        scaler: Pre-trained scaler (if any)
    
    Returns:
        Tuple containing pose errors, metrics, timings, and uncertainty information
    """
    # Create pipeline with scaling and model
    pipeline = make_pipeline(
        scaler,
        MultiOutputRegressor(
            BayesianRidge(
                compute_score=True,
                verbose=True  # Show convergence progress
            )
        )
    )

    # Hyperparameter search space
    paramDist = {
        'multioutputregressor__estimator__alpha_1': loguniform(1e-7, 1e-3),
        'multioutputregressor__estimator__alpha_2': loguniform(1e-7, 1e-3),
        'multioutputregressor__estimator__lambda_1': loguniform(1e-7, 1e-3),
        'multioutputregressor__estimator__lambda_2': loguniform(1e-7, 1e-3),
    }

    # Randomized search with cross-validation
    search = RandomizedSearchCV(
        pipeline,
        paramDist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )

    # Train with timing
    search.fit(XTrain, yTrain)

    # Get best model and predictions
    bestBR = search.best_estimator_
    trainingTime = search.cv_results_['mean_fit_time'][search.best_index_]

    yPred, testingTime = testModel(XTest, bestBR, scaler)

    # Get prediction uncertainties
    yStd = np.sqrt(np.array([
        estimator.predict(XTest, return_std=True)[1] 
        for estimator in bestBR.named_steps['multioutputregressor'].estimators_
    ])).T  # Shape: (n_samples, n_targets)

    # Metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    
    # Average uncertainty metrics
    avgStd = np.mean(yStd)
    maxStd = np.max(yStd)

    print(f"""
    === Bayesian Regression Results ===
    - Best Params: {search.best_params_}
    - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}
    - Avg Uncertainty: {avgStd:.4f}, Max Uncertainty: {maxStd:.4f}
    - Training Time: {trainingTime:.2f}s
    """)

    # Calculate pose errors with uncertainty
    pose_errors = calculatePoseErrors(yPred, yTest, robot)
    
    return pose_errors, mse, mae, trainingTime, testingTime, r2#, yStd  # Return uncertainties