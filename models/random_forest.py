# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import time

from generate.generate_data import calculatePoseErrors


def randomForest(XTrain, yTrain, XTest, yTest, robot, scaler):
    rf = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

    # Train the model
    startTrain = time.time()
    rf.fit(XTrain, yTrain)
    endTrain = time.time()
    trainingTime = endTrain - startTrain

    # Test the model
    startTest = time.time()
    yPred = scaler.inverse_transform(rf.predict(XTest))
    endTest = time.time()
    testingTime = endTest - startTest
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime