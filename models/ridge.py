# Ridge Regression Model
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from generate.generate_data import testModel, calculatePoseErrors, decodeAngles

from test.test import testFittingOnTrainingSet, testKFoldCV

import numpy as np


def ridgeRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a Ridge Regression model
    
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
    rrPipe = make_pipeline(
        StandardScaler(),
        Ridge(random_state=42)
    )
    
    # Define parameter grid
    paramGrid = {
        'ridge__alpha': np.logspace(-3, 3, 7)
    }

    # Perform grid search
    gridSearch = GridSearchCV(
        rrPipe, 
        paramGrid, 
        cv=3, 
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    gridSearch.fit(XTrain, yTrain)
    
    # Find the best model
    bestRR = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Perform k-Fold Cross-Validation
    testKFoldCV(XTrain, yTrain, bestRR, k=5, scaler=scaler, modelName="Ridge Regression")

    # Test the best model
    yPred, testingTime = testModel(XTest, bestRR, scaler)

    # Inverse transform the actual values to get the original scale
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
    testFittingOnTrainingSet(XTrain, bestRR, scaler, "Ridge Regression")
    
    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2, gridSearch.best_params_
