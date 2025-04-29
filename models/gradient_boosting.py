# Gradient Boosting Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import testModel, calculatePoseErrors, decodeAngles

from test.test import testFittingOnTrainingSet, testKFoldCV


def gradientBoosting(XTrain, yTrain, XTest, yTest, robot):
	"""
    Train and test a Gradient Boosting model

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
		- randomSearch.best_params_: Best parameters for the model
    """
	# Create pipeline
	gbPipe = make_pipeline(
		MultiOutputRegressor(GradientBoostingRegressor())
	)

	# Define parameter grid
	paramGrid = {
        'multioutputregressor__estimator__n_estimators': list(range(10, 110, 10)),
	    'multioutputregressor__estimator__max_depth': [3, 5, 10, 20, 50, None],  # Reduce search space
        'multioutputregressor__estimator__learning_rate': [0.001, 0.1, 0.01],
		'multioutputregressor__estimator__min_samples_split': [2, 5, 10],  # Split criteria
		'multioutputregressor__estimator__min_samples_leaf': [1, 5, 10],  # Min leaf size
	    'multioutputregressor__estimator__subsample': [0.5, 0.7, 1.0],  # Stochastic boosting
    	'multioutputregressor__estimator__max_features': [None, 'sqrt', 'log2'],  # Feature selection
    	'multioutputregressor__estimator__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],  # Different loss functions
		'multioutputregressor__estimator__random_state': [42],
	}

	# Perform grid search
	randomSearch = RandomizedSearchCV(
		gbPipe, 
		paramGrid, 
		cv=3, 
		n_jobs=-1,
		scoring='neg_mean_squared_error', 
	)
	randomSearch.fit(XTrain, yTrain)
	
	# Find the best model
	bestGB = randomSearch.best_estimator_
	trainingTime = randomSearch.cv_results_['mean_fit_time'][randomSearch.best_index_]

	# Perform k-Fold Cross-Validation
	testKFoldCV(XTrain, yTrain, bestGB, k=5, scaler=None, modelName="Gradient Boosting")
	
	# Test the best model
	yPred, testingTime = testModel(XTest, bestGB, None)  # No scaler used in this case

	# Decode angles to ensure equal weighting in distance calculations
	yTest = decodeAngles(yTest[:, :7], yTest[:, 7:])

	# Calculate metrics
	mse = mean_squared_error(yTest, yPred)
	mae = mean_absolute_error(yTest, yPred)
	r2 = r2_score(yTest, yPred)
	
	# Calculate pose errors
	poseErrors = calculatePoseErrors(yPred, yTest, robot)
	
	# VALIDATION - Perform fitting on the training set
	testFittingOnTrainingSet(XTrain, bestGB, None, "Gradient Boosting")  # No scaler needed for Gradient Boosting

	# Return results
	return poseErrors, mse, mae, trainingTime, testingTime, r2, randomSearch.best_params_