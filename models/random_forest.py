# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, decodeAngles

import time
import numpy as np


def randomForest(XTrain, yTrain, XTest, yTest, robot):
	"""
    Train and test a Random Forest model

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
	rfPipe = make_pipeline(
		RandomForestRegressor()
	)

	# Define parameter grid
	paramGrid = {
        'randomforestregressor__n_estimators': [50, 100, 200],
        'randomforestregressor__max_depth': [None, 10, 20],
        'randomforestregressor__min_samples_split': [2, 5, 10],
        'randomforestregressor__min_samples_leaf': [1, 2, 4]
	}

	# Perform grid search
	gridSearch = GridSearchCV(
		rfPipe, 
		paramGrid, 
		cv=3, 
		n_jobs=-1, 
		scoring='neg_mean_squared_error'
	)
	gridSearch.fit(XTrain, yTrain)
	
	# Find the best model
	bestRF = gridSearch.best_estimator_
	trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
	
	# Test the best model
	startTest = time.time()
	yPred = bestRF.predict(XTest)
	# Decode angles to ensure equal weighting in distance calculations
	yPred = decodeAngles(yPred[:, :7], yPred[:, 7:])
	endTest = time.time()
	testingTime = endTest - startTest

	# Decode angles to ensure equal weighting in distance calculations
	yTest = decodeAngles(yTest[:, :7], yTest[:, 7:])

	# Calculate metrics
	mse = mean_squared_error(yTest, yPred)
	mae = mean_absolute_error(yTest, yPred)
	r2 = r2_score(yTest, yPred)
	
	# Calculate pose errors
	poseErrors = calculatePoseErrors(yPred, yTest, robot)

	# VALIDATION - Perform fitting on the training set
	yPredTrain = bestRF.predict(XTrain)
	yPredTrainDecode = decodeAngles(yPredTrain[:, :7], yPredTrain[:, 7:])
	minPredTrain = np.min(yPredTrainDecode, axis=0)
	maxPredTrain = np.max(yPredTrainDecode, axis=0)
	print("Training set min:", minPredTrain)
	print("Training set max:", maxPredTrain)

	# Return results
	return poseErrors, mse, mae, trainingTime, testingTime, r2, gridSearch.best_params_