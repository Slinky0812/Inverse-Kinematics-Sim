# Kernel Ridge Regression Model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from generate.generate_data import calculatePoseErrors, testModel, decodeAngles

from test.test import testFittingOnTrainingSet, testKFoldCV


def kernelRidgeRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a Kernel Ridge Regression model

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
    krPipe = make_pipeline(
        StandardScaler(),
        KernelRidge()
    )

    # Define parameter grid
    paramGrid = {
        'kernelridge__alpha': [0.01, 0.1, 1.0, 10.0],
        'kernelridge__gamma': [0.01, 0.1, 1.0, 10.0],
        'kernelridge__kernel': ['rbf', 'poly', 'linear']
    }

    # Perform grid search
    gridSearch = GridSearchCV(
        krPipe, 
        paramGrid, 
        cv=3,
        n_jobs=-1, 
        scoring='neg_mean_squared_error'
    )
    gridSearch.fit(XTrain, yTrain)

    # Find the best model
    bestKR = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Perform k-Fold Cross-Validation
    testKFoldCV(XTrain, yTrain, bestKR, k=5, scaler=scaler, modelName="Kernel Ridge Regression")

    # Test the best model
    yPred, testingTime = testModel(XTest, bestKR, scaler)

    # Inverse transform the actual values to get them back to the original scale
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
    testFittingOnTrainingSet(XTrain, bestKR, scaler, "Kernel Ridge Regression")

    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2, gridSearch.best_params_