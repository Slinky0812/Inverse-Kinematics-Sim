import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plotData(kNNErrors, lRErrors, nNErrors):
    # Convert to NumPy arrays
    kNNErrors = np.array(kNNErrors)
    lRErrors = np.array(lRErrors)
    nNErrors = np.array(nNErrors)

    # Separate position and orientation errors
    kNNPosErrors, kNNOriErrors = kNNErrors[:, 0], kNNErrors[:, 1]
    lRPosErrors, lROriErrors = lRErrors[:, 0], lRErrors[:, 1]
    nNPosErrors, nNOriErrors = nNErrors[:, 0], nNErrors[:, 1]

    labels = ['kNN', 'Linear Regression', 'Neural Networks']

    # Plot position errors
    positionErrorsPlot(kNNPosErrors, lRPosErrors, nNPosErrors, labels)

    plt.show()


def positionErrorsPlot(kNNPosErrors, lRPosErrors, nNPosErrors, labels):
    # Prepare data for Seaborn
    data = {
        "Model": ["k-NN"] * len(kNNPosErrors) + ["Linear Regression"] * len(lRPosErrors) + ["Neural Network"] * len(nNPosErrors),
        "Position Error": np.concatenate([kNNPosErrors, lRPosErrors, nNPosErrors])
    }

    df = pd.DataFrame(data)

    # Plot violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="Model", y="Position Error", data=df)

    plt.title("Position Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)