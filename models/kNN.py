# k-Nearest Neighbors model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from generate.generate_data import calculatePoseErrors


def kNN(XTrain, yTrain, XTest, yTest, robot, scaler):
    # Train the model
    knn = KNeighborsRegressor(
        n_neighbors=5,  # Start with 5 neighbors
        weights='uniform',  # All neighbors contribute equally
        metric='euclidean'  # Standard distance metric
    )

    knn.fit(XTrain, yTrain)

    # Test the model
    yPred = scaler.inverse_transform(knn.predict(XTest))
    mse = mean_squared_error(yTest, yPred)
    mae = mean_absolute_error(yTest, yPred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Calculate pose errors
    poseErrors = calculatePoseErrors(yPred, XTest, robot)
    return poseErrors, mse, mae