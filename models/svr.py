# Support Vector Regression model
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from generate.generate_data import calculatePoseErrors, trainModel, testModel


def supportVectorRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    svr = SVR(kernel='rbf')
    multiSVR = MultiOutputRegressor(svr)

    # Train the model
    multiSVR, trainingTime = trainModel(XTrain, yTrain, multiSVR)

    # Test the model
    yPred, testingTime = testModel(XTest, yTest, multiSVR, scaler)
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime

