# Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from generate.generate_data import calculatePoseErrors, trainModel, testModel


def linearRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    lr = LinearRegression()

   # Train the model
    lr, trainingTime = trainModel(XTrain, yTrain, lr)

    # Test the model
    yPred, testingTime = testModel(XTest, lr, scaler)
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime


def bayesianLinearRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    br = BayesianRidge()
    multiBR = MultiOutputRegressor(br)

   # Train the model
    multiBR, trainingTime = trainModel(XTrain, yTrain, multiBR)

    # Test the model
    yPred, testingTime = testModel(XTest, multiBR, scaler)
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime