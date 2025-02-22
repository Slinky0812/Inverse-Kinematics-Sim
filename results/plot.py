import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plotErrorData(errors):
    # Convert to NumPy arrays
    kNNErrors = np.array(errors[0])
    lRErrors = np.array(errors[1])
    nNErrors = np.array(errors[2])
    dTErrors = np.array(errors[3])
    sVRErrors = np.array(errors[4])
    rFErrors = np.array(errors[5])
    gBErrors = np.array(errors[6])
    gRErrors = np.array(errors[7])
    bRErrors = np.array(errors[8])
    lassoErrors = np.array(errors[9])

    # Separate position and orientation errors
    kNNPosErrors, kNNOriErrors = kNNErrors[:, 0], kNNErrors[:, 1]
    lRPosErrors, lROriErrors = lRErrors[:, 0], lRErrors[:, 1]
    nNPosErrors, nNOriErrors = nNErrors[:, 0], nNErrors[:, 1]
    dTPosErrors, dTOriErrors = dTErrors[:, 0], dTErrors[:, 1]
    sVRPosErrors, sVROriErrors = sVRErrors[:, 0], sVRErrors[:, 1]
    rFPosErrors, rFOriErrors = rFErrors[:, 0], rFErrors[:, 1]
    gBPosErrors, gBOriErrors = gBErrors[:, 0], gBErrors[:, 1]
    gRPosErrors, gROriErrors = gRErrors[:, 0], gRErrors[:, 1]
    bRPosErrors, bROriErrors = bRErrors[:, 0], bRErrors[:, 1]
    lassoPosErrors, lassoOriErrors = lassoErrors[:, 0], lassoErrors[:, 1]

    # Plot position errors
    positionErrorsPlot(kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors, sVRPosErrors, rFPosErrors, gBPosErrors, gRPosErrors, bRPosErrors, lassoPosErrors)
    
    # Plot orientation errors
    orientationErrorsPlot(kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors, sVROriErrors, rFOriErrors, gBOriErrors, gROriErrors, bROriErrors, lassoOriErrors)


def positionErrorsPlot(kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors, sVRPosErrors, rFPosErrors, gBPosErrors, gRPosErrors, bRPosErrors, lassoPosErrors):
    # Prepare data for Seaborn
    data = {
        "Model": ["k-NN"] * len(kNNPosErrors) + 
                ["Linear Regression"] * len(lRPosErrors) + 
                ["Neural Network"] * len(nNPosErrors) + 
                ["Decision Trees"] * len(dTPosErrors) + 
                ["SVR"] * len(sVRPosErrors) + 
                ["Random Forest"] * len(rFPosErrors) + 
                ["Gradient Boosting"] * len(gBPosErrors) +
                ["Gaussian Process Regression"] * len(gRPosErrors) + 
                ["Bayesian Linear Regression"] * len(bRPosErrors) + 
                ["Lasso Regression"] * len(lassoPosErrors),
        "Position Error": np.concatenate([kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors, sVRPosErrors, rFPosErrors, gBPosErrors, gRPosErrors, bRPosErrors, lassoPosErrors])
    }

    df = pd.DataFrame(data)

    # Plot violin plot
    plt.figure(figsize=(25, 6))
    sns.violinplot(x="Model", y="Position Error", data=df)

    plt.title("Position Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("results/position_error_comparison.png")


def orientationErrorsPlot(kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors, sVROriErrors, rFOriErrors, gBOriErrors, gROriErrors, bROriErrors, lassoOriErrors):
    # Prepare data for Seaborn
    data = {
        "Model": ["k-NN"] * len(kNNOriErrors) + 
                ["Linear Regression"] * len(lROriErrors) + 
                ["Neural Network"] * len(nNOriErrors) + 
                ["Decision Trees"] * len(dTOriErrors) + 
                ["SVR"] * len(sVROriErrors) + 
                ["Random Forest"] * len(rFOriErrors) + 
                ["Gradient Boosting"] * len(gBOriErrors) + 
                ["Gaussian Process Regression"] * len(gROriErrors) + 
                ["Bayesian Linear Regression"] * len(bROriErrors) + 
                ["Lasso Regression"] * len(lassoOriErrors),
        "Orientation Error": np.concatenate([kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors, sVROriErrors, rFOriErrors, gBOriErrors, gROriErrors, bROriErrors, lassoOriErrors])
    }

    df = pd.DataFrame(data)

    # Plot violin plot
    plt.figure(figsize=(25, 6))
    sns.violinplot(x="Model", y="Orientation Error", data=df)

    plt.title("Orientation Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("results/orientation_error_comparison.png")


def plotMSEData(mseValues, models):
    plt.figure(figsize=(25, 6))

    # models = ["k-NN", "Linear Regression", "Neural Network", "Decision Trees", "SVR", "Random Forest", "Gradient Boosting", "Gaussian Process Regression", "Bayesian Linear Regression", "Lasso Regression"]

    plt.bar(models, mseValues)
    
    plt.xlabel("Models")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE Comparison of Different Models")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("results/mse_data_comparison.png")


def plotMAEData(maeValues, models):
    plt.figure(figsize=(25, 6))
    
    # models = ["k-NN", "Linear Regression", "Neural Network", "Decision Trees", "SVR", "Random Forest", "Gradient Boosting", "Gaussian Process Regression", "Bayesian Linear Regression", "Lasso Regression"]
    
    plt.bar(models, maeValues)
    
    plt.xlabel("Models")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE Comparison of Different Models")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("results/mae_data_comparison.png")


def plotTimings(trainingTimes, testingTimes, models):
    plt.figure(figsize=(25, 6))

    # models = ["k-NN", "Linear Regression", "Neural Network", "Decision Trees", "SVR", "Random Forest", "Gradient Boosting", "Gaussian Process Regression", "Bayesian Linear Regression", "Lasso Regression"]
    
    x = np.arange(len(models))  # X-axis positions

    # Create a grouped bar chart
    plt.bar(x - 0.2, trainingTimes, width=0.4, label="Training Time", color='blue')
    plt.bar(x + 0.2, testingTimes, width=0.4, label="Testing Time", color='orange')

    plt.xticks(x, models)
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Training vs. Testing Time for Models")
    plt.yscale("log")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("results/timings_comparison.png")