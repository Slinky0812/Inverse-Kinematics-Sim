from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from generate.generate_data import calculatePoseErrors


def decisionTree(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Train the model
    dt = DecisionTreeRegressor(
        random_state=42  # Random seed for reproducibility
    )
    dt.fit(XTrain, yTrain)

    # Test the model
    yPred = scaler.inverse_transform(dt.predict(XTest))
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae
