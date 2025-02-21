# Gaussian Process Regression model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, mean_absolute_error

import time

from generate.generate_data import calculatePoseErrors


def gaussianProcessRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Define the kernel (RBF kernel)
    kernel = 1.0 * RBF(length_scale=1.0)

    gp = GaussianProcessRegressor(kernel=kernel , n_restarts_optimizer=10)

    # Train the model
    startTrain = time.time()
    gp.fit(XTrain, yTrain)
    endTrain = time.time()
    trainingTime = endTrain - startTrain

    # Test the model
    startTest = time.time()
    yPred = scaler.inverse_transform(gp.predict(XTest))
    endTest = time.time()
    testingTime = endTest - startTest
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime