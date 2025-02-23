import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plotErrorData(errors, models):
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
    kRRErrors = np.array(errors[10])

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
    kRRPosErrors, kRROriErrors = kRRErrors[:, 0], kRRErrors[:, 1]


    # Create lists of all error arrays
    modelPosErrors = [
        kNNPosErrors, lRPosErrors, nNPosErrors, dTPosErrors,
        sVRPosErrors, rFPosErrors, gBPosErrors, gRPosErrors,
        bRPosErrors, lassoPosErrors, kRRPosErrors
    ]
    
    modelOriErrors = [
        kNNOriErrors, lROriErrors, nNOriErrors, dTOriErrors,
        sVROriErrors, rFOriErrors, gBOriErrors, gROriErrors,
        bROriErrors, lassoOriErrors, kRROriErrors
    ]

    # Plot position errors
    positionErrorsPlot(modelPosErrors, models)
    
    # Plot orientation errors
    orientationErrorsPlot(modelOriErrors, models)


def positionErrorsPlot(modelPosErrors, models):
    # Prepare data for Seaborn

    # Create lists of errors and corresponding model names
    errors_flat = np.concatenate(modelPosErrors)
    model_labels = np.repeat(models, [len(arr) for arr in modelPosErrors])
    
    df = pd.DataFrame({
        "Model": model_labels,
        "Position Error": errors_flat
    })

    # print("")
    # print("Position Errors")
    # print(f"kNN - {kNNPosErrors[199]}")
    # print(f"Linear Regression - {lRPosErrors[199]}")
    # print(f"Neural Networks - {nNPosErrors[199]}")
    # print(f"Decision Trees - {dTPosErrors[199]}")
    # print(f"Support Vector Regression - {sVRPosErrors[199]}")
    # print(f"Random Forest - {rFPosErrors[199]}")
    # print(f"Gradient Boosting - {gBPosErrors[199]}")
    # print(f"Gaussian Process Regression - {gRPosErrors[199]}")
    # print(f"Lasso Regression - {lassoPosErrors[199]}")
    # print(f"Kernel Ridge Regression - {kRRPosErrors[199]}")
    # print("")
    
    # Plot violin plot
    plt.figure(figsize=(25, 6))
    sns.violinplot(x="Model", y="Position Error", data=df)

    plt.title("Position Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("results/position_error_comparison.png")


def orientationErrorsPlot(modelOriErrors, models):
    # Prepare data for Seaborn
    errors_flat = np.concatenate(modelOriErrors)
    model_labels = np.repeat(models, [len(arr) for arr in modelOriErrors])
    
    df = pd.DataFrame({
        "Model": model_labels,
        "Orientation Error": errors_flat
    })

    # print("")
    # print("Orientation Errors")
    # print(f"kNN - {kNNOriErrors[199]}")
    # print(f"Linear Regression - {lROriErrors[199]}")
    # print(f"Neural Networks - {nNOriErrors[199]}")
    # print(f"Decision Trees - {dTOriErrors[199]}")
    # print(f"Support Vector Regression - {sVROriErrors[199]}")
    # print(f"Random Forest - {rFOriErrors[199]}")
    # print(f"Gradient Boosting - {gBOriErrors[199]}")
    # print(f"Gaussian Process Regression - {gROriErrors[199]}")
    # print(f"Lasso Regression - {lassoOriErrors[199]}")
    # print(f"Kernel Ridge Regression - {kRROriErrors[199]}")
    # print("")

    # Plot violin plot
    plt.figure(figsize=(25, 6))
    sns.violinplot(x="Model", y="Orientation Error", data=df)

    plt.title("Orientation Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig("results/orientation_error_comparison.png")


def plotMSEData(mseValues, models):
    plt.figure(figsize=(25, 6))

    plt.bar(models, mseValues)
    
    plt.xlabel("Models")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE Comparison of Different Models")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("results/mse_data_comparison.png")


def plotMAEData(maeValues, models):
    plt.figure(figsize=(25, 6))
        
    plt.bar(models, maeValues)
    
    plt.xlabel("Models")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE Comparison of Different Models")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("results/mae_data_comparison.png")


def plotTimings(trainingTimes, testingTimes, models):
    plt.figure(figsize=(25, 6))
    
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