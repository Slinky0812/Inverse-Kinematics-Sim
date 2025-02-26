# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, trainModel, testModel


def randomForest(XTrain, yTrain, XTest, yTest, robot, scaler):
	rfPipe = make_pipeline(
		scaler,
		RandomForestRegressor()
	)

	paramGrid = {
        'randomforestregressor__n_estimators': [50, 100, 200],
        'randomforestregressor__max_depth': [None, 10, 20],
        'randomforestregressor__min_samples_split': [2, 5, 10],
        'randomforestregressor__min_samples_leaf': [1, 2, 4]
	}

	gridSearch = GridSearchCV(rfPipe, paramGrid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
	gridSearch.fit(XTrain, yTrain)
	
	# Use the best model
	bestRF = gridSearch.best_estimator_
	trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
	
	# Test the model
	yPred, testingTime = testModel(XTest, bestRF, scaler)
	
	mse = mean_squared_error(yTest, yPred)
	mae = mean_absolute_error(yTest, yPred)
	r2 = r2_score(yTest, yPred)
	print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
	
	# Calculate pose errors
	poseErrors = calculatePoseErrors(yPred, XTest, robot)
	return poseErrors, mse, mae, trainingTime, testingTime, r2