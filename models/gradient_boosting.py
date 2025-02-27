# Gradient Boosting Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, trainModel, testModel


def gradientBoosting(XTrain, yTrain, XTest, yTest, robot, scaler):

	gbPipe = make_pipeline(
		MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
	)

	paramGrid = {
        'multioutputregressor__estimator__n_estimators': [50, 100, 200],
        'multioutputregressor__estimator__max_depth': [None, 10, 20],
        'multioutputregressor__estimator__learning_rate': [0.1, 0.01, 0.001],
	}

	gridSearch = RandomizedSearchCV(gbPipe, paramGrid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
	gridSearch.fit(XTrain, yTrain)
	
	# Use the best model
	bestGB = gridSearch.best_estimator_
	trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
	
	# Test the model
	yPred, testingTime = testModel(XTest, bestGB, scaler)

	mse = mean_squared_error(yTest, yPred)
	mae = mean_absolute_error(yTest, yPred)
	r2 = r2_score(yTest, yPred)
	print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
	
	# Calculate pose errors
	poseErrors = calculatePoseErrors(yPred, yTest, robot)
	return poseErrors, mse, mae, trainingTime, testingTime, r2