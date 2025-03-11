# Gaussian Process Regression model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, RationalQuadratic, Matern
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate.generate_data import calculatePoseErrors, testModel


def gaussianProcessRegression(XTrain, yTrain, XTest, yTest, robot, scaler):
    """
    Train and test a Gaussian Process Regression model

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
    # Define a list of candidate kernels
    kernel_options = [
        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * 
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + 
        WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-2, 1e2)),  # RBF

        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * 
        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + 
        WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-2, 1e2)),  # Matern

        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * 
        RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-2, 1e2)) + 
        WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-2, 1e2)),  # RationalQuadratic

        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * 
        (RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + 
        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)) + 
        WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-2, 1e2))  # RBF + Matern
    ]
        
    # Initialize the model
    gp = GaussianProcessRegressor(normalize_y=True)

    paramGrid = {
        "kernel": kernel_options,
        "alpha": [1e-10, 1e-5, 1e-2],  # Regularization parameter
        "n_restarts_optimizer": [5, 10, 20, 30, 50],  # Restarts for better optimization
    }

    
    # Perform grid search
    gridSearch = GridSearchCV(
        estimator=gp,
        param_grid=paramGrid,
        cv=3,
        n_jobs=2,
        scoring='neg_mean_squared_error',
    )
    gridSearch.fit(XTrain, yTrain)
    
    # Find the best model
    bestGP = gridSearch.best_estimator_
    trainingTime = gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]
    
    # Test the best model
    yPred, testingTime = testModel(XTest, bestGP, scaler)
    
    # Calculate metrics
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    
    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, yTest, robot)

    # Return results
    return poseErrors, mse, mae, trainingTime, testingTime, r2