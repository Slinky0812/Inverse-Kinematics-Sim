from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from generate.generate_data import testModel, decodeAngles

import numpy as np


def testFittingOnTrainingSet(XTrain, model, scaler, modelName):
    """
    Fit the model on the training set and print the min and max values of the predicted output

    Args:
        - XTrain (np.array): Training input set
        - yTrain (np.array): Training output set
        - model: ML model
        - scaler (StandardScaler): Scales the predicted values to match the scale of the actual values
    """
    # Perform fitting on the training set
    yPred, _ = testModel(XTrain, model, scaler) # Testing time is not needed

    minPred = np.min(yPred, axis=0)
    maxPred = np.max(yPred, axis=0)

    # Open file in append mode
    with open('results/training_set_min_and_max.txt', 'a') as f:
        f.write(f"{modelName}:\n")
        f.write(f"Min: {minPred}\n")
        f.write(f"Max: {maxPred}\n\n")


def testKFoldCV(X, y, model, k, scaler, modelName):
    """
    Perform K-Fold Cross Validation on the model and print the mean and standard deviation of the MSE
    
    Args:
        - X (np.array): Input set
        - y (np.array): Output set
        - model: ML model
        - k (int): Number of folds for K-Fold Cross Validation
        - scaler (StandardScaler): Scales the predicted values to match the scale of the actual values
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mseValues = []

    for trainIndex, testIndex in kf.split(X):
        XTrain, XTest = X[trainIndex], X[testIndex]
        yTrain, yTest = y[trainIndex], y[testIndex]

        # Fit the model on the training set
        model.fit(XTrain, yTrain)

        # Test the model on the test set
        yPred, _ = testModel(XTest, model, scaler)

        if scaler is not None:
            # Inverse transform the actual values to get them back to the original scale
            yTest = scaler.inverse_transform(yTest)
        
        # Decode angles to ensure equal weighting in distance calculations
        yTestDecode = decodeAngles(yTest[:, :7], yTest[:, 7:])

        # Calculate MSE
        mse = mean_squared_error(yTestDecode, yPred)
        mseValues.append(mse)

    meanMSE = np.mean(mseValues)
    stdMSE = np.std(mseValues)

    print(f"Mean MSE: {meanMSE}, Std MSE: {stdMSE}")

    # Open file in write mode
    with open('results/k_fold_cv_results.txt', 'a') as f:
        f.write(f"{modelName}:\n")
        f.write(f"Mean MSE: {meanMSE}, Std MSE: {stdMSE}\n\n")