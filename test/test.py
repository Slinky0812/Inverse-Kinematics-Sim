import numpy as np

from generate.generate_data import testModel


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
    yPred, testingTime = testModel(XTrain, model, scaler) # Testing time is not needed

    minPred = np.min(yPred, axis=0)
    maxPred = np.max(yPred, axis=0)

    # Open file in append mode
    with open('results/training_set_min_and_max.txt', 'a') as f:
        f.write(f"{modelName}:\n")
        f.write(f"Min: {minPred}\n")
        f.write(f"Max: {maxPred}\n\n")