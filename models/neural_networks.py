# Neural Network model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import computePoseErrors, testModel

import time

def neuralNetwork(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a Neural Network model

    Args:
        - XTrain (np.array): Training input set
        - yTrain (np.array): Training output set
        - XTest (np.array): Testing input set
        - yTest (np.array): Testing output set
        - robot (RobotController): Robot object
        - scaler (StandardScalar): Scaler object

    Returns:
        - poseErrors (np.array): Array of position and orientation errors
        - mse (float): Mean Squared Error
        - mae (float): Mean Absolute Error
        - trainingTime (float): Training time
        - testingTime (float): Testing time
        - r2 (float): R² score
    """
    # Create pipeline
    nNPipe = make_pipeline(
        scaler,
        MLPRegressor(warm_start=True)
    )
    
    # Define Parameter grid
    paramGrid = {
        'mlpregressor__hidden_layer_sizes': [
            (256, 256), (512, 256, 128),  # Varying depths
            (128, 128), (512, 512)
        ],
        'mlpregressor__activation': ['relu', 'tanh'],
        'mlpregressor__solver': ['adam', 'sgd'],  # Test different solvers
        'mlpregressor__max_iter': [5000],
        'mlpregressor__early_stopping': [True],
        'mlpregressor__validation_fraction': [0.15],  # Slightly more validation data
        'mlpregressor__n_iter_no_change': [25],  # Longer patience
        'mlpregressor__learning_rate': ['adaptive'],
        'mlpregressor__alpha': [0.0001, 0.001, 0.01],
        'mlpregressor__random_state': [42]
    }

    # Perform grid search
    gridSearch = GridSearchCV(
        nNPipe,
        paramGrid,
        cv=3,  # Faster than default 5-fold
        n_jobs=-1,  # Use all CPU cores
        scoring='neg_mean_squared_error'  # Focus on MSE during search
    )
    gridSearch.fit(XTrain, yTrain)

    # Find the best model
    bestNN = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
    
    # Test the best model
    yPred, testingTime = testModel(XTest, bestNN, scaler)

    # Calculate metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    # Calculate pose errors
    poseErrors = computePoseErrors(yPred, yTest, robot)

    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2