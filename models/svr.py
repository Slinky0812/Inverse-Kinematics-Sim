# Support Vector Regression model
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, trainModel, testModel

import time


def supportVectorRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    svrPipe = make_pipeline(
        scaler,
        MultiOutputRegressor(LinearSVR(max_iter=1000000))
    )

    # Define parameter grid
    paramGrid = {
        'multioutputregressor__estimator__C': [0.1, 1, 10, 100],
        'multioutputregressor__estimator__epsilon': [0.01, 0.1, 0.5],
    }

    # Perform grid search
    gridSearch = GridSearchCV(svrPipe, paramGrid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    gridSearch.fit(XTrain, yTrain)
    
    # Use the best model
    bestMultiSVR = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Test the model
    yPred, testingTime = testModel(XTest, bestMultiSVR, scaler)

    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2