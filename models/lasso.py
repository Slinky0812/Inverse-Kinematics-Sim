# Lasso Regression model
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, trainModel, testModel

def lassoRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Initialize Lasso model with default parameters
    lasso = Lasso(
        alpha=1.0,  # Regularization strength
        max_iter=1000,  # Increased iterations for convergence
        random_state=42  # Seed for reproducibility (if data shuffling is used)
    )

    # Train the model
    lasso, trainingTime = trainModel(XTrain, yTrain, lasso)

    # Test the model
    yPred, testingTime = testModel(XTest, lasso, scaler)
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2