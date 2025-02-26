from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pybullet_controller import RobotController
from generate.generate_data import generateIKDataset
from results.plot import plotErrorData, plotMSEData, plotMAEData, plotTimings, plotR2Score

from models.kNN import kNN
from models.linear_regression import linearRegression, bayesianLinearRegression
from models.neural_networks import neuralNetwork
from models.decision_trees import decisionTree
from models.svr import supportVectorRegression
from models.random_forest import randomForest
from models.gradient_boosting import gradientBoosting
from models.gaussian_regression import gaussianProcessRegression
from models.lasso import lassoRegression
from models.kernel_ridge import kernelRidgeRegression


def main():
    # Create instance of robot controller
    robot = RobotController()
    robot.createWorld(view_world=False)

    # Generate data set
    # X = the end effector pose
    # y = the joint angles
    X, y = generateIKDataset(robot, numSamples=1000)

    # Split data into training and testing sets (80% training, 20% testing)
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalise features to ensure equal weighting in distance calculations:
    scaler = StandardScaler()
    XTrainScaled = scaler.fit_transform(XTrain)
    XTestScaled = scaler.transform(XTest)

    yTrainScaled = scaler.fit_transform(yTrain)
    yTestScaled = scaler.transform(yTest)

    # lists to keep track of errors, timings, mae and mse values
    models = []
    errors = []
    mseValues = []
    maeValues = []
    trainingTimes = []
    testingTimes = []
    r2Scores = []

    print("")
    print("kNN")
    # train the model using k-Nearest Neighbors
    kNNErrors, kNNmse, kNNmae, kNNTrainingTime, kNNTestingTime, kNNr2 = kNN(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("k-Nearest Neighbours")
    errors.append(kNNErrors)
    mseValues.append(kNNmse)
    maeValues.append(kNNmae)
    trainingTimes.append(kNNTrainingTime)
    testingTimes.append(kNNTestingTime)
    r2Scores.append(kNNr2)
    
    print("")
    print("Linear Regression")
    # train the model using Linear Regression
    lRErrors, lRmse, lRmae, lRTrainingTime, lRTestingTime, lRr2 = linearRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Linear Regression")
    errors.append(lRErrors)
    mseValues.append(lRmse)
    maeValues.append(lRmae)
    trainingTimes.append(lRTrainingTime)
    testingTimes.append(lRTestingTime)
    r2Scores.append(lRr2)
    
    # print("")
    # print("Neural Networks")
    # # train the model using Neural Networks
    # nNErrors, nNmse, nNmae, nNTrainingTime, nNTestingTime, nNr2 = neuralNetwork(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    # models.append("Neural Networks")
    # errors.append(nNErrors)
    # mseValues.append(nNmse)
    # maeValues.append(nNmae)
    # trainingTimes.append(nNTrainingTime)
    # testingTimes.append(nNTestingTime)
    # r2Scores.append(nNr2)
    
    print("")
    print("Decision Trees")
    # train the model using Neural Networks
    dTErrors, dTmse, dTmae, dTTrainingTime, dTTestingTime, dTr2 = decisionTree(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Decision Trees")
    errors.append(dTErrors)
    mseValues.append(dTmse)
    maeValues.append(dTmae)
    trainingTimes.append(dTTrainingTime)
    testingTimes.append(dTTestingTime)
    r2Scores.append(dTr2)

    print("")
    print("Support Vector Regression")
    # train the model using Support Vector Regression
    sVRErrors, sVRmse, sVRmae, sVRTrainingTime, sVRTestingTime, sVRr2 = supportVectorRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Support Vector Regression")
    errors.append(sVRErrors)
    mseValues.append(sVRmse)
    maeValues.append(sVRmae)
    trainingTimes.append(sVRTrainingTime)
    testingTimes.append(sVRTestingTime)
    r2Scores.append(sVRr2)

    print("")
    print("Random Forest")
    # train the model using Random Forest
    rFErrors, rFmse, rFmae, rFTrainingTime, rFTestingTime, rFr2 = randomForest(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Random Forest")
    errors.append(rFErrors)
    mseValues.append(rFmse)
    maeValues.append(rFmae)
    trainingTimes.append(rFTrainingTime)
    testingTimes.append(rFTestingTime)
    r2Scores.append(rFr2)

    print("")
    print("Gradient Boosting")
    # train the model using Gradient Boosting
    gBErrors, gBmse, gBmae, gBTrainingTime, gBTestingTime, gBr2 = gradientBoosting(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Gradient Boosting")
    errors.append(gBErrors)
    mseValues.append(gBmse)
    maeValues.append(gBmae)
    trainingTimes.append(gBTrainingTime)
    testingTimes.append(gBTestingTime)
    r2Scores.append(gBr2)

    print("")
    print("Gaussian Process Regression")
    # train the model using Gaussian Process Regression
    gRErrors, gRmse, gRmae, gRTrainingTime, gRTestingTime, gRr2 = gaussianProcessRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Gaussian Process Regression")
    errors.append(gRErrors)
    mseValues.append(gRmse)
    maeValues.append(gRmae)
    trainingTimes.append(gRTrainingTime)
    testingTimes.append(gRTestingTime)
    r2Scores.append(gRr2)

    print("")
    print("Bayesian Linear Regression")
    # train the model using Bayesian Linear Regression
    bRErrors, bRmse, bRmae, bRTrainingTime, bRTestingTime, bRr2 = bayesianLinearRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Bayesian Linear Regression")
    errors.append(bRErrors)
    mseValues.append(bRmse)
    maeValues.append(bRmae)
    trainingTimes.append(bRTrainingTime)
    testingTimes.append(bRTestingTime)
    r2Scores.append(bRr2)

    print("")
    print("Lasso Regression")
    # train the model using Lasso Regression
    lassoErrors, lassoMSE, lassoMAE, lassoTrainingTime, lassoTestingTime, lassoR2 = lassoRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Lasso Regression")
    errors.append(lassoErrors)
    mseValues.append(lassoMSE)
    maeValues.append(lassoMAE)
    trainingTimes.append(lassoTrainingTime)
    testingTimes.append(lassoTestingTime)
    r2Scores.append(lassoR2)

    print("")
    print("Kernel Ridge Regression")
    # train the model using Lasso Regression
    kRRErrors, kRRmse, kRRmae, kRRTrainingTime, kRRTestingTime, kRRr2 = kernelRidgeRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Kernel Ridge Regression")
    errors.append(kRRErrors)
    mseValues.append(kRRmse)
    maeValues.append(kRRmae)
    trainingTimes.append(kRRTrainingTime)
    testingTimes.append(kRRTestingTime)
    r2Scores.append(kRRr2)

    # plot the results
    plotErrorData(errors, models)
    plotMSEData(mseValues, models)
    plotMAEData(maeValues, models)
    plotTimings(trainingTimes, testingTimes, models)
    plotR2Score(r2Scores, models)

    # print("\n")
    # print("Difference between Linear and Bayes:")
    # print(f"MSE - Linear: {lRmse}    Bayes: {bRmse}")
    # print(f"MAE - Linear: {lRmae}    Bayes: {bRmae}")
    # print(f"Training Times - Linear: {lRTrainingTime}    Bayes: {bRTrainingTime}")
    # print(f"Testing Times - Linear: {lRTestingTime}    Bayes: {bRTestingTime}")
    # print("\n")
    

if __name__ == "__main__":
    main()