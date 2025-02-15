import pybullet as p
import time
import pybullet_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pybullet_controller import RobotController
from generate.generate_data import generateIKDataset
from results.plot import plotData

from models.kNN import kNN
from models.linear_regression import linearRegression
from models.neural_networks import neuralNetwork

def main():
    # Create instance of robot controller
    robot = RobotController()
    robot.createWorld(view_world=False)

    # Generate data set
    # X = the end effector pose
    # y = the joint angles
    X, y = generateIKDataset(robot, num_samples=1000)

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

    yTrainScaled = scaler.fit_transform(yTrain)
    yTestScaled = scaler.transform(yTest)

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

    print("")
    print("kNN")
    # train the model using k-Nearest Neighbors
    kNNErrors, kNNmse, kNNmae, kNNTrainingTime, kNNTestingTime = kNN(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    for i in range(len(kNNErrors)):
        print("")
        print(kNNErrors[i])
    
    print("")
    print("Linear Regression")
    # train the model using Linear Regression
    lRErrors, lRmse, lRmae, lRTrainingTime, lRTestingTime = linearRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    for i in range(len(lRErrors)):
        print("")
        print(lRErrors[i])

    print("")
    print("Neural Networks")
    # train the model using Neural Networks
    nNErrors, nNmse, nNmae, nNTrainingTime, nNTestingTime = neuralNetwork(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    for i in range(len(nNErrors)):
        print("")
        print(nNErrors[i])

    print("")
    print("Decision Trees")
    # train the model using Neural Networks
    dTErrors, dTmse, dTmae, dTTrainingTime, dTTestingTime = neuralNetwork(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    for i in range(len(dTErrors)):
        print("")
        print(dTErrors[i])


    # plot the errors
    # plot_error_distribution(kNNErrors)
    plotData(kNNErrors, lRErrors, nNErrors, dTErrors)

    print("Training times:")
    print(f"kNN: {kNNTrainingTime:.4f} seconds")
    print(f"Linear Regression: {lRTrainingTime:.4f} seconds")
    print(f"Neural Networks: {nNTrainingTime:.4f} seconds")
    print(f"Decision Trees: {dTTrainingTime:.4f} seconds")
    print("")

    print("Testing times:")
    print(f"kNN: {kNNTestingTime:.4f} seconds")
    print(f"Linear Regression: {lRTestingTime:.4f} seconds")
    print(f"Neural Networks: {nNTestingTime:.4f} seconds")
    print(f"Decision Trees: {dTTestingTime:.4f} seconds")


if __name__ == "__main__":
    main()