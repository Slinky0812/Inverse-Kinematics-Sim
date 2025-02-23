# Gaussian Process Regression model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, trainModel, testModel


def gaussianProcessRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Define the kernel (RBF kernel)
    kernel = 1.0 * RBF(length_scale=1.0)

    gp = GaussianProcessRegressor(kernel=kernel , n_restarts_optimizer=10)

    # Train the model
    gp, trainingTime = trainModel(XTrain, yTrain, gp)

    # Test the model
    yPred, testingTime = testModel(XTest, gp, scaler)
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2