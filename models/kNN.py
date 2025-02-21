# k-Nearest Neighbors model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

import time

from generate.generate_data import calculatePoseErrors


def kNN(XTrain, yTrain, XTest, yTest, robot, scaler):
    knn = KNeighborsRegressor()
    # Define the range of k values to test
    param_grid = {'n_neighbors': list(range(1, 21))}  # Test k from 1 to 20

    # Perform cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(XTrain, yTrain)

    # Get the best k value
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Optimal k: {best_k}")

    bestKNN = KNeighborsRegressor(
        n_neighbors=best_k,  # Start with 5 neighbors
    )

    # Train the model
    startTrain = time.time()
    bestKNN.fit(XTrain, yTrain)
    endTrain = time.time()
    trainingTime = endTrain - startTrain

    # Test the model
    startTest = time.time()
    yPred = scaler.inverse_transform(bestKNN.predict(XTest))
    endTest = time.time()
    testingTime = endTest - startTest
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime