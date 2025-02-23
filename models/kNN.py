# k-Nearest Neighbors model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, testModel


def kNN(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Create pipeline with scaling
    knnPipe = make_pipeline(
        scaler,
        KNeighborsRegressor()
    )
    
    # Expanded parameter grid
    paramGrid = {
        'kneighborsregressor__n_neighbors': list(range(1, 31, 2)),
        'kneighborsregressor__weights': ['uniform', 'distance'],
        'kneighborsregressor__p': [1, 2]
    }
    
    # Multi-metric grid search
    gridSearch = GridSearchCV(knnPipe, paramGrid, cv=5,
                             scoring={'MSE': 'neg_mean_squared_error',
                                      'MAE': 'neg_mean_absolute_error'},
                             refit='MSE', n_jobs=-1)
    
    gridSearch.fit(XTrain, yTrain)
    # Get best model and metrics
    bestKNN = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
    
    # Test the model
    # yPred = bestKNN.predict(XTest)
    # testingTime = time.time() - start_time  # Implement timing
    yPred, testingTime = testModel(XTest, bestKNN, scaler)

    # Calculate metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    
    print(f"Best Params: {gridSearch.best_params_}")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}") #, R²: {r2:.4f}")

    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2