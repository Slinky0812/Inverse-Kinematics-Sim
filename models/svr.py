# Support Vector Regression model
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, trainModel, testModel


def supportVectorRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    # svr = SVR(kernel='rbf')
    # multiSVR = MultiOutputRegressor(svr)

    svrPipe = make_pipeline(
        scaler,
        MultiOutputRegressor(SVR())
    )

    # Define parameter grid
    paramGrid = {
        'multioutputregressor__estimator__C': [0.1, 1, 10, 100],
        'multioutputregressor__estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'multioutputregressor__estimator__epsilon': [0.01, 0.1, 0.5],
        'multioutputregressor__estimator__kernel': ['linear', 'poly', 'rbf']
    }

    # Perform grid search
    gridSearch = GridSearchCV(svrPipe, paramGrid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    print("training")
    gridSearch.fit(XTrain, yTrain)
    print("all done :)")
    # Use the best model
    bestMultiSVR = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]

    # Train the model
    # multiSVR, trainingTime = trainModel(XTrain, yTrain, multiSVR)

    # Test the model
    yPred, testingTime = testModel(XTest, bestMultiSVR, scaler)

    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2