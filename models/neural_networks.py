# Neural Network model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, testModel

import time

def neuralNetwork(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Create pipeline with proper naming
    nNPipe = make_pipeline(
        scaler,
        MLPRegressor()
    )
    
    # Parameter grid
    paramGrid = {
        'mlpregressor__hidden_layer_sizes': [
            (256, 256), (512, 256, 128),  # Varying depths
            (256, 256, 256), (512, 512)
        ],
        'mlpregressor__activation': ['relu', 'tanh', 'relu'],
        'mlpregressor__solver': ['adam', 'lbfgs'],  # Test different solvers
        'mlpregressor__max_iter': [2000],
        'mlpregressor__early_stopping': [True],
        'mlpregressor__validation_fraction': [0.15],  # Slightly more validation data
        'mlpregressor__n_iter_no_change': [25],  # Longer patience
        'mlpregressor__learning_rate': ['adaptive'],
        'mlpregressor__alpha': [0.0001, 0.001, 0.01],
        'mlpregressor__random_state': [42]
    }

    # Optimized GridSearch setup
    gridSearch = GridSearchCV(
        nNPipe,
        paramGrid,
        cv=3,  # Faster than default 5-fold
        n_jobs=-1,  # Use all CPU cores
        scoring='neg_mean_squared_error'  # Focus on MSE during search
    )

    # Timing with proper benchmarking
    currentTime = time.time()
    gridSearch.fit(XTrain, yTrain)
    endTime = time.time() - currentTime
    print(f"Total time finding best model = {endTime}")

    # Get best model
    bestNN = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
    
    # Test the model
    yPred, testingTime = testModel(XTest, bestNN, scaler)

    # Metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2