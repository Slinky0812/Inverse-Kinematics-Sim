import pybullet as p
import time
import pybullet_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from pybullet_controller import RobotController
from generate.generate_data import generateIKDataset
from models.kNN import kNN
from results.plot import plot_error_distribution

def main():
    # Create instance of robot controller
    robot = RobotController()
    robot.createWorld(view_world=False)

    # Generate data set
    # X = the end effector pose
    # y = the joint angles
    X, y = generateIKDataset(robot, num_samples=10)

    # print("")
    # for i in range(len(X)):
    #     print(f"X: {X[i]}")
    #     print(f"y: {y[i]}")
    #     print("")

    # Split data into training and testing sets (80% training, 20% testing)
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalise features to ensure equal weighting in distance calculations:
    scaler = StandardScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)

    # print("")
    # for i in range(len(X_train_scaled)):
    #     print(f"X train: {X_train_scaled[i]}")
    #     print(f"y train: {y_train[i]}")
    #     print("")

    # print("")
    # for i in range(len(X_test_scaled)):
    #     print(f"X test: {X_test_scaled[i]}")
    #     print(f"y test: {y_test[i]}")
    #     print("")

    # train the model using k-Nearest Neighbors
    kNNErrors = kNN(XTrainScaled, yTrain, XTestScaled, yTest, robot)
    for i in range(len(kNNErrors)):
        print(kNNErrors[i])
        print("")

    # plot the errors
    plot_error_distribution(kNNErrors)


if __name__ == "__main__":
    main()