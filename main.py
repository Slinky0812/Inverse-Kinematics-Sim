from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pybullet_controller import RobotController
from generate.generate_data import generateIKDataset
from results.plot import plotErrorData, plotMSEData, plotMAEData, plotTimings

from models.kNN import kNN
from models.linear_regression import linearRegression, bayesianLinearRegression
from models.neural_networks import neuralNetwork
from models.decision_trees import decisionTree
from models.svr import supportVectorRegression
from models.random_forest import randomForest
from models.gradient_boosting import gradientBoosting
from models.gaussian_regression import gaussianProcessRegression
from models.lasso import lassoRegression


def main():
    # Create instance of robot controller
    robot = RobotController()
    robot.createWorld(view_world=False)

    # Generate data set
    # X = the end effector pose
    # y = the joint angles
    X, y = generateIKDataset(robot, num_samples=1000)

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

    print("")
    print("kNN")
    # train the model using k-Nearest Neighbors
    kNNErrors, kNNmse, kNNmae, kNNTrainingTime, kNNTestingTime = kNN(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("kNN")
    errors.append(kNNErrors)
    mseValues.append(kNNmse)
    maeValues.append(kNNmae)
    trainingTimes.append(kNNTrainingTime)
    testingTimes.append(kNNTestingTime)
    
    print("")
    print("Linear Regression")
    # train the model using Linear Regression
    lRErrors, lRmse, lRmae, lRTrainingTime, lRTestingTime = linearRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Linear Regression")
    errors.append(lRErrors)
    mseValues.append(lRmse)
    maeValues.append(lRmae)
    trainingTimes.append(lRTrainingTime)
    testingTimes.append(lRTestingTime)
    
    print("")
    print("Neural Networks")
    # train the model using Neural Networks
    nNErrors, nNmse, nNmae, nNTrainingTime, nNTestingTime = neuralNetwork(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Neural Networks")
    errors.append(nNErrors)
    mseValues.append(nNmse)
    maeValues.append(nNmae)
    trainingTimes.append(nNTrainingTime)
    testingTimes.append(nNTestingTime)
    
    print("")
    print("Decision Trees")
    # train the model using Neural Networks
    dTErrors, dTmse, dTmae, dTTrainingTime, dTTestingTime = decisionTree(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Decision Trees")
    errors.append(dTErrors)
    mseValues.append(dTmse)
    maeValues.append(dTmae)
    trainingTimes.append(dTTrainingTime)
    testingTimes.append(dTTestingTime)

    print("")
    print("Support Vector Regression")
    # train the model using Support Vector Regression
    sVRErrors, sVRmse, sVRmae, sVRTrainingTime, sVRTestingTime = supportVectorRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Support Vector Regression")
    errors.append(sVRErrors)
    mseValues.append(sVRmse)
    maeValues.append(sVRmae)
    trainingTimes.append(sVRTrainingTime)
    testingTimes.append(sVRTestingTime)

    print("")
    print("Random Forest")
    # train the model using Random Forest
    rFErrors, rFmse, rFmae, rFTrainingTime, rFTestingTime = randomForest(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Random Forest")
    errors.append(rFErrors)
    mseValues.append(rFmse)
    maeValues.append(rFmae)
    trainingTimes.append(rFTrainingTime)
    testingTimes.append(rFTestingTime)

    print("")
    print("Gradient Boosting")
    # train the model using Gradient Boosting
    gBErrors, gBmse, gBmae, gBTrainingTime, gBTestingTime = gradientBoosting(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Gradient Boosting")
    errors.append(gBErrors)
    mseValues.append(gBmse)
    maeValues.append(gBmae)
    trainingTimes.append(gBTrainingTime)
    testingTimes.append(gBTestingTime)

    print("")
    print("Gaussian Process Regression")
    # train the model using Gaussian Process Regression
    gRErrors, gRmse, gRmae, gRTrainingTime, gRTestingTime = gaussianProcessRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Gaussian Process Regression")
    errors.append(gRErrors)
    mseValues.append(gRmse)
    maeValues.append(gRmae)
    trainingTimes.append(gRTrainingTime)
    testingTimes.append(gRTestingTime)

    print("")
    print("Bayesian Linear Regression")
    # train the model using Bayesian Linear Regression
    bRErrors, bRmse, bRmae, bRTrainingTime, bRTestingTime = bayesianLinearRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Bayesian Linear Regression")
    errors.append(bRErrors)
    mseValues.append(bRmse)
    maeValues.append(bRmae)
    trainingTimes.append(bRTrainingTime)
    testingTimes.append(bRTestingTime)

    print("")
    print("Lasso Regression")
    # train the model using Lasso Regression
    lassoErrors, lassoMSE, lassoMAE, lassoTrainingTime, lassoTestingTime = lassoRegression(XTrainScaled, yTrainScaled, XTestScaled, yTestScaled, robot, scaler)
    models.append("Lasso Regression")
    errors.append(lassoErrors)
    mseValues.append(lassoMSE)
    maeValues.append(lassoMAE)
    trainingTimes.append(lassoTrainingTime)
    testingTimes.append(lassoTestingTime)

    # plot the results
    plotErrorData(errors)
    plotMSEData(mseValues, models)
    plotMAEData(maeValues, models)
    plotTimings(trainingTimes, testingTimes, models)

    # print("\n")
    # print("Difference between Linear and Bayes:")
    # print(f"MSE - Linear: {lRmse}    Bayes: {bRmse}")
    # print(f"MAE - Linear: {lRmae}    Bayes: {bRmae}")
    # print(f"Training Times - Linear: {lRTrainingTime}    Bayes: {bRTrainingTime}")
    # print(f"Testing Times - Linear: {lRTestingTime}    Bayes: {bRTestingTime}")
    # print("\n")
    

if __name__ == "__main__":
    main()