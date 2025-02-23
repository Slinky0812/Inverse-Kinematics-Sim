# Kernel Ridge Regression model
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, trainModel, testModel

def kernelRidgeRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    kr = KernelRidge(
        alpha=1.0,          # Regularization strength
        kernel='rbf',       # Radial Basis Function kernel
        gamma=None,         # Automatically uses 1/n_features
        degree=3,           # Only used for polynomial kernel
        coef0=1             # Bias term for polynomial kernel
    )

    # Train the model
    kr, trainingTime = trainModel(XTrain, yTrain, kr)

    # Test the model
    yPred, testingTime = testModel(XTest, kr, scaler)
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2