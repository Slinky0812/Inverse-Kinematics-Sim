import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pybullet_kinematics_dynamics_control.pybullet_controller import RobotController
from results.plot import plotErrorData, plotMSEData, plotMAEData, plotTimings, plotR2Score, storeBestParams

from generate.generate_data import encodeAngles, decodeAngles

from models.kNN import kNN
from models.linear_regression import linearRegression, bayesianLinearRegression
from models.neural_networks import neuralNetwork
from models.decision_trees import decisionTree
from models.svr import supportVectorRegression
from models.random_forest import randomForest
from models.gradient_boosting import gradientBoosting
from models.gaussian_regression import gaussianProcessRegression
from models.lasso import lassoRegression
from models.ridge import ridgeRegression
from models.kernel_ridge import kernelRidgeRegression


def main():
    """
    Main function to run the inverse kinematics calculations on different models and evaluate the results
    """
    # Create instance of robot controller
    robot = RobotController(robot_type='/PandaRobot.jl/deps/Panda/panda', controllable_joints=[0, 1, 2, 3, 4, 5, 6])
    robot.createWorld(view_world=False)

    # Load data set
    # X = the end effector pose
    # y = the joint angles
    with open('robot-trajectory.pkl', mode='rb') as f:
        dataset = pickle.load(f)

    inputData = []
    outputData = []
    for data in dataset:
        inputData.append(data[:, :3])
        outputData.append(data[:, 3:])

    X = np.vstack(inputData)  # Shape: (Total_N, 3)
    y = np.vstack(outputData) # Shape: (Total_N, 7)

    # Encode angles to ensure equal weighting in distance calculations
    y = encodeAngles(y)  # Shape: (Total_N, 14)

    print("Input shape:", X.shape)
    print("Output shape:", y.shape)

    # Split data into training and testing sets (80% training, 20% testing)
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    # VALIDATION - Print the min and max values of the joint angles in the training set
    yTrainDecoded = decodeAngles(yTrain[:, :7], yTrain[:, 7:])
    with open('results/training_set_min_and_max.txt', 'w') as f:
        f.write("yTrain:\n")
        f.write(f"Min: {np.min(yTrainDecoded, axis=0)}\n")
        f.write(f"Max: {np.max(yTrainDecoded, axis=0)}\n\n")

    # Normalise features to ensure equal weighting in distance calculations
    yScaler = StandardScaler()

    yTrainScaled = yScaler.fit_transform(yTrain)
    yTestScaled = yScaler.transform(yTest)

    # Create lists to keep track of models, errors, timings, mae and mse values, and R² scores
    models = []
    errors = []
    mseValues = []
    maeValues = []
    trainingTimes = []
    testingTimes = []
    r2Scores = []
    bestParams = []

    print("")
    print("kNN")
    # Solve the IK problem using k-Nearest Neighbors
    kNNErrors, kNNmse, kNNmae, kNNTrainingTime, kNNTestingTime, kNNr2, kNNBestParams = kNN(XTrain, yTrainScaled, XTest, yTestScaled, robot, yScaler)
    models.append("kNN")
    errors.append(kNNErrors)
    mseValues.append(kNNmse)
    maeValues.append(kNNmae)
    trainingTimes.append(kNNTrainingTime)
    testingTimes.append(kNNTestingTime)
    r2Scores.append(kNNr2)
    bestParams.append(kNNBestParams)
    
    print("")
    print("Linear Regression")
    # Solve the IK problem using Linear Regression
    lRErrors, lRmse, lRmae, lRTrainingTime, lRTestingTime, lRr2, lRBestParams = linearRegression(XTrain, yTrainScaled, XTest, yTestScaled, robot, yScaler)
    models.append("Linear Regression")
    errors.append(lRErrors)
    mseValues.append(lRmse)
    maeValues.append(lRmae)
    trainingTimes.append(lRTrainingTime)
    testingTimes.append(lRTestingTime)
    r2Scores.append(lRr2)
    bestParams.append(lRBestParams)
    
    print("")
    print("Neural Networks")
    # Solve the IK problem using Neural Networks
    nNErrors, nNmse, nNmae, nNTrainingTime, nNTestingTime, nNr2, nNBestParams = neuralNetwork(XTrain, yTrainScaled, XTest, yTestScaled, robot, yScaler)
    models.append("Neural Networks")
    errors.append(nNErrors)
    mseValues.append(nNmse)
    maeValues.append(nNmae)
    trainingTimes.append(nNTrainingTime)
    testingTimes.append(nNTestingTime)
    r2Scores.append(nNr2)
    bestParams.append(nNBestParams)

    print("")
    print("Decision Trees")
    # Solve the IK problem using Decision Trees
    dTErrors, dTmse, dTmae, dTTrainingTime, dTTestingTime, dTr2, dTBestParams = decisionTree(XTrain, yTrain, XTest, yTest, robot)
    models.append("Decision Trees")
    errors.append(dTErrors)
    mseValues.append(dTmse)
    maeValues.append(dTmae)
    trainingTimes.append(dTTrainingTime)
    testingTimes.append(dTTestingTime)
    r2Scores.append(dTr2)
    bestParams.append(dTBestParams)

    print("")
    print("Support Vector Regression")
    # Solve the IK problem using Support Vector Regression
    sVRErrors, sVRmse, sVRmae, sVRTrainingTime, sVRTestingTime, sVRr2, sVRBestParams = supportVectorRegression(XTrain, yTrainScaled, XTest, yTestScaled, robot, yScaler)
    models.append("SVR")
    errors.append(sVRErrors)
    mseValues.append(sVRmse)
    maeValues.append(sVRmae)
    trainingTimes.append(sVRTrainingTime)
    testingTimes.append(sVRTestingTime)
    r2Scores.append(sVRr2)
    bestParams.append(sVRBestParams)

    print("")
    print("Random Forest")
    # Solve the IK problem using Random Forest
    rFErrors, rFmse, rFmae, rFTrainingTime, rFTestingTime, rFr2, rFBestParams = randomForest(XTrain, yTrain, XTest, yTest, robot)
    models.append("Random Forest")
    errors.append(rFErrors)
    mseValues.append(rFmse)
    maeValues.append(rFmae)
    trainingTimes.append(rFTrainingTime)
    testingTimes.append(rFTestingTime)
    r2Scores.append(rFr2)
    bestParams.append(rFBestParams)

    print("")
    print("Gradient Boosting")
    # Solve the IK problem using Gradient Boosting
    gBErrors, gBmse, gBmae, gBTrainingTime, gBTestingTime, gBr2, gBBestParams = gradientBoosting(XTrain, yTrain, XTest, yTest, robot)
    models.append("Gradient Boosting")
    errors.append(gBErrors)
    mseValues.append(gBmse)
    maeValues.append(gBmae)
    trainingTimes.append(gBTrainingTime)
    testingTimes.append(gBTestingTime)
    r2Scores.append(gBr2)
    bestParams.append(gBBestParams)

    print("")
    print("Gaussian Process Regression")
    # Solve the IK problem using Gaussian Process Regression
    gRErrors, gRmse, gRmae, gRTrainingTime, gRTestingTime, gRr2, gRBestParams = gaussianProcessRegression(XTrain, yTrainScaled, XTest, yTestScaled, robot, yScaler)
    models.append("GPR")
    errors.append(gRErrors)
    mseValues.append(gRmse)
    maeValues.append(gRmae)
    trainingTimes.append(gRTrainingTime)
    testingTimes.append(gRTestingTime)
    r2Scores.append(gRr2)
    bestParams.append(gRBestParams)

    print("")
    print("Bayesian Linear Regression")
    # Solve the IK problem using Bayesian Linear Regression
    bRErrors, bRmse, bRmae, bRTrainingTime, bRTestingTime, bRr2, bRBestParams = bayesianLinearRegression(XTrain, yTrainScaled, XTest, yTestScaled, robot, yScaler)
    models.append("BLR")
    errors.append(bRErrors)
    mseValues.append(bRmse)
    maeValues.append(bRmae)
    trainingTimes.append(bRTrainingTime)
    testingTimes.append(bRTestingTime)
    r2Scores.append(bRr2)
    bestParams.append(bRBestParams)

    print("")
    print("Lasso Regression")
    # Solve the IK problem using Lasso Regression
    lassoErrors, lassoMSE, lassoMAE, lassoTrainingTime, lassoTestingTime, lassoR2, lassoBestParams = lassoRegression(XTrain, yTrainScaled, XTest, yTestScaled, robot, yScaler)
    models.append("Lasso Regression")
    errors.append(lassoErrors)
    mseValues.append(lassoMSE)
    maeValues.append(lassoMAE)
    trainingTimes.append(lassoTrainingTime)
    testingTimes.append(lassoTestingTime)
    r2Scores.append(lassoR2)
    bestParams.append(lassoBestParams)

    print("")
    print("Ridge Regression")
    # Solve the IK problem using Ridge Regression
    rRErrors, rRmse, rRmae, rRTrainingTime, rRTestingTime, rRr2, rRBestParams = ridgeRegression(XTrain, yTrainScaled, XTest, yTestScaled, robot, yScaler)
    models.append("Ridge Regression")
    errors.append(rRErrors)
    mseValues.append(rRmse)
    maeValues.append(rRmae)
    trainingTimes.append(rRTrainingTime)
    testingTimes.append(rRTestingTime)
    r2Scores.append(rRr2)
    bestParams.append(rRBestParams)

    print("")
    print("Kernel Ridge Regression")
    # Solve the IK problem using Kernel Ridge Regression
    kRRErrors, kRRmse, kRRmae, kRRTrainingTime, kRRTestingTime, kRRr2, kRRBestParams = kernelRidgeRegression(XTrain, yTrainScaled, XTest, yTestScaled, robot, yScaler)
    models.append("KRR")
    errors.append(kRRErrors)
    mseValues.append(kRRmse)
    maeValues.append(kRRmae)
    trainingTimes.append(kRRTrainingTime)
    testingTimes.append(kRRTestingTime)
    r2Scores.append(kRRr2)
    bestParams.append(kRRBestParams)

    print("")

    # Plot the results
    plotErrorData(errors, models)
    plotMSEData(mseValues, models)
    plotMAEData(maeValues, models)
    plotTimings(trainingTimes, testingTimes, models)
    plotR2Score(r2Scores, models)
    storeBestParams(bestParams, models)


if __name__ == "__main__":
    main()