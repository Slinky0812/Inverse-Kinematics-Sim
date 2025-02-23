# Decision Tree model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, trainModel, testModel


def decisionTree(XTrain, yTrain, XTest, yTest, robot, scaler):
    dt = DecisionTreeRegressor(
        random_state=42  # Random seed for reproducibility
    )

    # Train the model
    dt, trainingTime = trainModel(XTrain, yTrain, dt)

    # Test the model
    yPred, testingTime = testModel(XTest, dt, scaler)
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2
