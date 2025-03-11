from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, testModel


def decisionTree(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a Decision Tree model

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
    dtPipe = make_pipeline(
        DecisionTreeRegressor(random_state=42)
    )

    # Define parameter grid
    paramGrid = {
        'decisiontreeregressor__max_depth': [None, 5, 10, 15, 20],
        'decisiontreeregressor__min_samples_split': [2, 5, 10],
        'decisiontreeregressor__min_samples_leaf': [1, 2, 4],
        'decisiontreeregressor__max_features': ['sqrt', 'log2', None],
        'decisiontreeregressor__ccp_alpha': [0.0, 0.01, 0.1]
    }

    # Perform grid search
    gridSearch = GridSearchCV(
        dtPipe,
        paramGrid,
        cv=3,
        n_jobs=2,
        scoring='neg_mean_squared_error',
    )
    # with parallel_backend('loky'):  # Use loky backend
    gridSearch.fit(XTrain, yTrain)

    # Find the best model
    bestDT = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Test the best model
    yPred, testingTime = testModel(XTest, bestDT, scaler)
    
    # Calculate metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)

    # Pose errors
    poseErrors = calculatePoseErrors(yPred, yTest, robot)
    
    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2