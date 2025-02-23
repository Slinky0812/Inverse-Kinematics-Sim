# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from generate.generate_data import calculatePoseErrors, trainModel, testModel


def randomForest(XTrain, yTrain, XTest, yTest, robot, scaler):
    rf = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

      # Train the model
    rf, trainingTime = trainModel(XTrain, yTrain, rf)

    # Test the model
    yPred, testingTime = testModel(XTest, rf, scaler)

    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime