# Gradient Boosting Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from generate.generate_data import calculatePoseErrors, trainModel, testModel


def gradientBoosting(XTrain, yTrain, XTest, yTest, robot, scaler):
    gb = GradientBoostingRegressor(
        n_estimators=100,   # Number of boosting stages
        learning_rate=0.1,  # Step size shrinkage (trade-off between speed & accuracy)
        max_depth=3,        # Depth of each tree (controls complexity)
        random_state=42
    )
    multiGB = MultiOutputRegressor(gb)

   # Train the model
    multiGB, trainingTime = trainModel(XTrain, yTrain, multiGB)

    # Test the model
    yPred, testingTime = testModel(XTest, yTest, multiGB, scaler)

    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime