# Neural Network model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import time

from generate.generate_data import calculatePoseErrors


def neuralNetwork(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Train the model
    nn = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128),  # 2 hidden layers with 100 neurons each
        activation='relu',  # Rectified Linear Unit activation function
        solver='adam',  # Adaptive Moment Estimation optimization algorithm
        max_iter=1000,  # Maximum number of iterations
        warm_start=True,  # Reuse the solution of the previous call to fit as initialization
        random_state=42  # Random seed for reproducibility
    )
    startTrain = time.time()
    nn.fit(XTrain, yTrain)
    endTrain = time.time()
    trainingTime = endTrain - startTrain

    # Test the model
    startTest = time.time()
    yPred = scaler.inverse_transform(nn.predict(XTest))
    endTest = time.time()
    testingTime = endTest - startTest

    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime
