# Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import time

from generate.generate_data import calculatePoseErrors


def linearRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Train the model
    lr = LinearRegression()
    startTrain = time.time()
    lr.fit(XTrain, yTrain)
    endTrain = time.time()
    trainingTime = endTrain - startTrain

    # Test the model
    startTest = time.time()
    yPred = scaler.inverse_transform(lr.predict(XTest))
    endTest = time.time()
    testingTime = endTest - startTest
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime

