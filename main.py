from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pybullet_controller import RobotController
from generate.generate_data import generateIKDataset
from results.plot import plotErrorData, plotMSEData, plotMAEData, plotTimings

from models.kNN import kNN
from models.linear_regression import linearRegression
from models.neural_networks import neuralNetwork
from models.decision_trees import decisionTree
from models.svr import supportVectorRegression
from models.random_forest import randomForest

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

    errors = []
    mseValues = []
    maeValues = []
    trainingTimes = []
    testingTimes = []

    print("")
    print("kNN")
    # train the model using k-Nearest Neighbors
    kNNErrors, kNNmse, kNNmae, kNNTrainingTime, kNNTestingTime = kNN(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    errors.append(kNNErrors)
    mseValues.append(kNNmse)
    maeValues.append(kNNmae)
    trainingTimes.append(kNNTrainingTime)
    testingTimes.append(kNNTestingTime)
    # for i in range(len(kNNErrors)):
    #     print("")
    #     print(kNNErrors[i])
    
    print("")
    print("Linear Regression")
    # train the model using Linear Regression
    lRErrors, lRmse, lRmae, lRTrainingTime, lRTestingTime = linearRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    errors.append(lRErrors)
    mseValues.append(lRmse)
    maeValues.append(lRmae)
    trainingTimes.append(lRTrainingTime)
    testingTimes.append(lRTestingTime)
    # for i in range(len(lRErrors)):
    #     print("")
    #     print(lRErrors[i])

    print("")
    print("Neural Networks")
    # train the model using Neural Networks
    nNErrors, nNmse, nNmae, nNTrainingTime, nNTestingTime = neuralNetwork(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    errors.append(nNErrors)
    mseValues.append(nNmse)
    maeValues.append(nNmae)
    trainingTimes.append(nNTrainingTime)
    testingTimes.append(nNTestingTime)
    # for i in range(len(nNErrors)):
    #     print("")
    #     print(nNErrors[i])

    print("")
    print("Decision Trees")
    # train the model using Neural Networks
    dTErrors, dTmse, dTmae, dTTrainingTime, dTTestingTime = decisionTree(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    errors.append(dTErrors)
    mseValues.append(dTmse)
    maeValues.append(dTmae)
    trainingTimes.append(dTTrainingTime)
    testingTimes.append(dTTestingTime)
    # for i in range(len(dTErrors)):
    #     print("")
    #     print(dTErrors[i])


    print("")
    print("Support Vector Regression")
    # train the model using Support Vector Regression
    sVRErrors, sVRmse, sVRmae, sVRTrainingTime, sVRTestingTime = supportVectorRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    errors.append(sVRErrors)
    mseValues.append(sVRmse)
    maeValues.append(sVRmae)
    trainingTimes.append(sVRTrainingTime)
    testingTimes.append(sVRTestingTime)
    # for i in range(len(sVRErrors)):
    #     print("")
    #     print(sVRErrors[i])

    print("")
    print("Random Forest")
    # train the model using Support Vector Regression
    rFErrors, rFmse, rFmae, rFTrainingTime, rFTestingTime = supportVectorRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    errors.append(rFErrors)
    mseValues.append(rFmse)
    maeValues.append(rFmae)
    trainingTimes.append(rFTrainingTime)
    testingTimes.append(rFTestingTime)


    # plot the results
    plotErrorData(errors)
    plotMSEData(mseValues)
    plotMAEData(maeValues)
    plotTimings(trainingTimes, testingTimes)

    print("Training times:")
    print(f"kNN: {kNNTrainingTime:.4f} seconds")
    print(f"Linear Regression: {lRTrainingTime:.4f} seconds")
    print(f"Neural Networks: {nNTrainingTime:.4f} seconds")
    print(f"Decision Trees: {dTTrainingTime:.4f} seconds")
    print(f"Support Vector Regression: {sVRTrainingTime:.4f} seconds")
    print("")

    print("Testing times:")
    print(f"kNN: {kNNTestingTime:.4f} seconds")
    print(f"Linear Regression: {lRTestingTime:.4f} seconds")
    print(f"Neural Networks: {nNTestingTime:.4f} seconds")
    print(f"Decision Trees: {dTTestingTime:.4f} seconds")
    print(f"Support Vector Regression: {sVRTestingTime:.4f} seconds")
    print("")

    # print("MSE:")
    # print(f"kNN: {kNNmse:.4f}")
    # print(f"Linear Regression: {lRmse:.4f}")
    # print(f"Neural Networks: {nNmse:.4f}")
    # print(f"Decision Trees: {dTmse:.4f}")
    # print(f"Support Vector Regression: {sVRmse:.4f}")
    # print("")

    # print("MAE:")
    # print(f"kNN: {kNNmae:.4f}")
    # print(f"Linear Regression: {lRmae:.4f}")
    # print(f"Neural Networks: {nNmae:.4f}")
    # print(f"Decision Trees: {dTmae:.4f}")
    # print(f"Support Vector Regression: {sVRmae:.4f}")
    # print("")



if __name__ == "__main__":
    main()