# Support Vector Regression model
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import time
import numpy as np

from generate.generate_data import calculatePoseErrors


def supportVectorRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    svr = SVR(kernel='rbf')
    multiSVR = MultiOutputRegressor(svr)

    # Train the model
    startTrain = time.time()
    multiSVR.fit(XTrain, yTrain)
    endTrain = time.time()
    trainingTime = endTrain - startTrain

    # Test the model
    startTest = time.time()
    yPred = scaler.inverse_transform(multiSVR.predict(XTest))
    endTest = time.time()
    testingTime = endTest - startTest
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime

