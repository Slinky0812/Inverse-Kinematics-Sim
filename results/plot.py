import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plotData(kNNErrors, lRErrors, nNErrors, dTErrors):
    # Convert to NumPy arrays
    kNNErrors = np.array(kNNErrors)
    lRErrors = np.array(lRErrors)
    nNErrors = np.array(nNErrors)
    dTErrors = np.array(dTErrors)

    # Separate position and orientation errors
    kNNPosErrors, kNNOriErrors = kNNErrors[:, 0], kNNErrors[:, 1]
    lRPosErrors, lROriErrors = lRErrors[:, 0], lRErrors[:, 1]
    nNPosErrors, nNOriErrors = nNErrors[:, 0], nNErrors[:, 1]
    dTPosErrors, dTOriErrors = dTErrors[:, 0], dTErrors[:, 1]

    labels = ['kNN', 'Linear Regression', 'Neural Networks', 'Decision Trees']

    # Plot position errors
    positionErrorsPlot(kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors, labels)
    orientationErrorsPlot(kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors, labels)

    plt.show()


def positionErrorsPlot(kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors, labels):
    # Prepare data for Seaborn
    data = {
        "Model": ["k-NN"] * len(kNNPosErrors) + ["Linear Regression"] * len(lRPosErrors) + ["Neural Network"] * len(nNPosErrors) + ["Decision Trees"] * len(dTPosErrors),
        "Position Error": np.concatenate([kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors])
    }

    df = pd.DataFrame(data)

    # Plot violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="Model", y="Position Error", data=df)

    plt.title("Position Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)


def orientationErrorsPlot(kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors, labels):
    # Prepare data for Seaborn
    data = {
        "Model": ["k-NN"] * len(kNNOriErrors) + ["Linear Regression"] * len(lROriErrors) + ["Neural Network"] * len(nNOriErrors) + ["Decision Trees"] * len(dTOriErrors),
        "Orientation Error": np.concatenate([kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors])
    }

    df = pd.DataFrame(data)

    # Plot violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="Model", y="Orientation Error", data=df)

    plt.title("Orientation Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)
