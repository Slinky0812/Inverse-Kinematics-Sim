# k-Nearest Neighbors model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, testModel


def kNN(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a k-Nearest Neighbours model

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
    knnPipe = make_pipeline(
        scaler,
        KNeighborsRegressor()
    )
    
    # define parameter grid
    paramGrid = {
        'kneighborsregressor__n_neighbors': list(range(1, 31, 2)),
        'kneighborsregressor__weights': ['uniform', 'distance'],
        'kneighborsregressor__p': [1, 2]
    }
    
    # Perform grid search
    gridSearch = GridSearchCV(
        knnPipe, 
        paramGrid, 
        cv=3,
        n_jobs=2,
        scoring='neg_mean_squared_error',
        refit='MSE', 
    )
    gridSearch.fit(XTrain, yTrain)

    # Find the best model
    bestKNN = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
    
    # Test the best model
    yPred, testingTime = testModel(XTest, bestKNN, scaler)

    # Calculate metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    # Pose errors
    poseErrors = calculatePoseErrors(yPred, yTest, robot)

    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2