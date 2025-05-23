# Support Vector Regression Model
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from generate.generate_data import calculatePoseErrors, testModel, decodeAngles

from test.test import testFittingOnTrainingSet, testKFoldCV


def supportVectorRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a Support Vector Regression model

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
        - gridSearch.best_params_: Best parameters for the model
    """
    # Create pipeline
    svrPipe = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(LinearSVR(max_iter=1000000))
    )

    # Define parameter grid
    paramGrid = {
        'multioutputregressor__estimator__C': [0.1, 1, 10, 100],
        'multioutputregressor__estimator__epsilon': [0.01, 0.1, 0.5],
    }

    # Perform grid search
    gridSearch = GridSearchCV(
        svrPipe, 
        paramGrid, 
        cv=3, 
        n_jobs=-1, 
        scoring='neg_mean_squared_error'
    )
    gridSearch.fit(XTrain, yTrain)
    
    # Find the best model
    bestSVR = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Perform k-Fold Cross-Validation
    testKFoldCV(XTrain, yTrain, bestSVR, k=5, scaler=scaler, modelName="Support Vector Regression")

    # Test the best model
    yPred, testingTime = testModel(XTest, bestSVR, scaler)

    # Inverse transform the actual values to get the original scale
    yTestScaled = scaler.inverse_transform(yTest)
    # Decode angles to ensure equal weighting in distance calculations
    yTestDecode = decodeAngles(yTestScaled[:, :7], yTestScaled[:, 7:])

    # Calculate metrics
    mse = mean_squared_error(yTestDecode, yPred)
    mae = mean_absolute_error(yTestDecode, yPred)
    r2 = r2_score(yTestDecode, yPred)

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, yTestDecode, robot)

    # VALIDATION - Perform fitting on the training set
    testFittingOnTrainingSet(XTrain, bestSVR, scaler, "Support Vector Regression")

    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2, gridSearch.best_params_