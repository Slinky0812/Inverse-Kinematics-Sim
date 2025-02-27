# Kernel Ridge Regression model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, trainModel, testModel

def kernelRidgeRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    krPipe = make_pipeline(
        scaler,
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
        krPipe, paramGrid, cv=5, 
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    gridSearch.fit(XTrain, yTrain)

    bestKR = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
    
    # Test the model
    yPred, testingTime = testModel(XTest, bestKR, scaler)
    
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, yTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2