# Gradient Boosting Model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, testModel


def gradientBoosting(XTrain, yTrain, XTest, yTest, robot, scaler):
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
    """
	# Create pipeline
	gbPipe = make_pipeline(
		MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
	)

	# Define parameter grid
	paramGrid = {
        'multioutputregressor__estimator__n_estimators': [50, 100, 200],
        'multioutputregressor__estimator__max_depth': [None, 10, 20],
        'multioutputregressor__estimator__learning_rate': [0.1, 0.01, 0.001],
	}

	# Perform grid search
	gridSearch = RandomizedSearchCV(
		gbPipe, 
		paramGrid, 
		cv=3, 
		n_jobs=2,
		scoring='neg_mean_squared_error', 
	)
	gridSearch.fit(XTrain, yTrain)
	
	# Find the best model
	bestGB = gridSearch.best_estimator_
	trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
	
	# Test the best model
	yPred, testingTime = testModel(XTest, bestGB, scaler)

	# Calculate metrics
	mse = mean_squared_error(yTest, yPred)
	mae = mean_absolute_error(yTest, yPred)
	r2 = r2_score(yTest, yPred)
	
	# Calculate pose errors
	poseErrors = calculatePoseErrors(yPred, yTest, robot)

	# Return results
	return poseErrors, mse, mae, trainingTime, testingTime, r2