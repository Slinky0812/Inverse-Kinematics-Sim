from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import parallel_backend

from generate.generate_data import calculatePoseErrors, testModel


def decisionTree(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Create pipeline without scaler
    dtPipe = Pipeline([
        ('decisiontreeregressor', DecisionTreeRegressor(random_state=42))
    ])

    # Enhanced parameter grid
    paramGrid = {
        'decisiontreeregressor__max_depth': [None, 5, 10, 15, 20],
        'decisiontreeregressor__min_samples_split': [2, 5, 10],
        'decisiontreeregressor__min_samples_leaf': [1, 2, 4],
        'decisiontreeregressor__max_features': ['sqrt', 'log2', None],
        'decisiontreeregressor__ccp_alpha': [0.0, 0.01, 0.1]
    }

    # Configure grid search
    gridSearch = GridSearchCV(
        dtPipe,
        paramGrid,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        refit=True
    )

    # Time training properly
    with parallel_backend('loky'):  # Use loky backend
        gridSearch.fit(XTrain, yTrain)

    # Get best model
    bestDT = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
    
    # Inspect best parameters
    print(f"Best parameters: {gridSearch.best_params_}")

    # Test the model (assuming testModel is updated)
    yPred, testingTime = testModel(XTest, bestDT, scaler)
    
    # Metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Feature importance analysis
    print("Feature importances:", 
          bestDT.named_steps['decisiontreeregressor'].feature_importances_)

    # Pose errors
    poseErrors = calculatePoseErrors(yPred, yTest, robot)
    return poseErrors, mse, mae, trainingTime, testingTime, r2