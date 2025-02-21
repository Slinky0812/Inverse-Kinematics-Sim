import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plotData(kNNErrors, lRErrors, nNErrors, dTErrors, sVRErrors):
    # Convert to NumPy arrays
    kNNErrors = np.array(kNNErrors)
    lRErrors = np.array(lRErrors)
    nNErrors = np.array(nNErrors)
    dTErrors = np.array(dTErrors)
    sVRErrors = np.array(sVRErrors)

    # Separate position and orientation errors
    kNNPosErrors, kNNOriErrors = kNNErrors[:, 0], kNNErrors[:, 1]
    lRPosErrors, lROriErrors = lRErrors[:, 0], lRErrors[:, 1]
    nNPosErrors, nNOriErrors = nNErrors[:, 0], nNErrors[:, 1]
    dTPosErrors, dTOriErrors = dTErrors[:, 0], dTErrors[:, 1]
    sVRPosErrors, sVROriErrors = sVRErrors[:, 0], sVRErrors[:, 1]

    # Plot position errors
    positionErrorsPlot(kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors, sVRPosErrors)
    orientationErrorsPlot(kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors, sVROriErrors)


def positionErrorsPlot(kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors, sVRPosErrors):
    # Prepare data for Seaborn
    data = {
        "Model": ["k-NN"] * len(kNNPosErrors) + ["Linear Regression"] * len(lRPosErrors) + ["Neural Network"] * len(nNPosErrors) + ["Decision Trees"] * len(dTPosErrors) + ["Support Vector Regression"] * len(sVRPosErrors),
        "Position Error": np.concatenate([kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors, sVRPosErrors])
    }

    df = pd.DataFrame(data)

    # Plot violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Model", y="Position Error", data=df)

    plt.title("Position Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("results/position_error_comparison.png")


def orientationErrorsPlot(kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors, sVROriErrors):
    # Prepare data for Seaborn
    data = {
        "Model": ["k-NN"] * len(kNNOriErrors) + ["Linear Regression"] * len(lROriErrors) + ["Neural Network"] * len(nNOriErrors) + ["Decision Trees"] * len(dTOriErrors) + ["Support Vector Regression"] * len(sVROriErrors),
        "Orientation Error": np.concatenate([kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors, sVROriErrors])
    }

    df = pd.DataFrame(data)

    # Plot violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Model", y="Orientation Error", data=df)

    plt.title("Orientation Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("results/orientation_error_comparison.png")
