from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from generate.generate_data import processEndEffectorPoses


def kNN(X_train, y_train, X_test, y_test, robot):
    # train the model
    knn = KNeighborsRegressor(
        n_neighbors=5,  # Start with 5 neighbors
        weights='uniform',  # All neighbors contribute equally
        metric='euclidean'  # Standard distance metric
    )

    knn.fit(X_train, y_train)

    # test the model
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    pose_errors = []
    for angles_pred, target_pose in zip(y_pred, X_test):
        # Compute achieved pose via forward kinematics
        achieved_pose = robot.solveForwardPositonKinematics(angles_pred)
        # Calculate position/orientation error
        achieved_poseProcessed = processEndEffectorPoses(achieved_pose)
        error = achieved_poseProcessed - target_pose
        pose_errors.append(error)

    return pose_errors