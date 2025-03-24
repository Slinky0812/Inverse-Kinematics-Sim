import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plotErrorData(errors, models):
    """
    Find position and orientation errors for each model

    Args:
        - errors (np.array): A 2D array of errors for each model, with the first entry being position errors and the second being orientation errors
        - models (array): A list of model names
    """
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

    # Plot position vs orientation errors
    positionVsOrientation(modelPosErrors, modelOriErrors, models)


def positionErrorsPlot(modelPosErrors, models):
    """
    Plot position errors for each model in a violin plot

    Args:
        - modelPosErrors (np.array): List of position errors for each model
        - models (array): List of model names
    """
    # Create lists of errors and corresponding model names
    errorsFlat = np.concatenate(modelPosErrors)
    modelLabels = np.repeat(models, [len(arr) for arr in modelPosErrors])
    
    df = pd.DataFrame({
        "Model": modelLabels,
        "Position Error": errorsFlat
    })
    
    # Plot violin plot
    plt.figure(figsize=(30, 6))
    sns.violinplot(x="Model", y="Position Error", data=df)

    plt.title("Position Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save plot
    plt.savefig("results/position_error_comparison.png")


def orientationErrorsPlot(modelOriErrors, models):
    """
    Plot oritentation errors for each model in a violin plot

    Args:
        - modelOriErrors (np.array): List of orientation errors for each model
        - models (array): List of model names
    """
    # Prepare data for Seaborn
    errors_flat = np.concatenate(modelOriErrors)
    model_labels = np.repeat(models, [len(arr) for arr in modelOriErrors])
    
    df = pd.DataFrame({
        "Model": model_labels,
        "Orientation Error": errors_flat
    })

    # Plot violin plot
    plt.figure(figsize=(30, 6))
    sns.violinplot(x="Model", y="Orientation Error", data=df)

    plt.title("Orientation Error Comparison (Violin Plot)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save plot
    plt.savefig("results/orientation_error_comparison.png")


def positionVsOrientation(modelPosErrors, modelOriErrors, models):
    """
    Plot position error vs orientation error for each model as a scatter plot

    Args:
        - modelPosErrors (np.array): List of position errors for each model
        - modelOriErrors (np.array): List of orientation errors for each model
        - models (array): List of model names
    """
    # Loop through each model
    for i, model in enumerate(models):
        plt.figure(figsize=(15, 10))

        # Scatter plot
        plt.scatter(modelPosErrors[i], modelOriErrors[i], s=100)

        # Fit a line of best fit (1st degree polynomial)
        coefficients = np.polyfit(modelPosErrors[i], modelOriErrors[i], 1)
        trendline = np.poly1d(coefficients)

        # Plot the line of best fit
        plt.plot(
            modelPosErrors[i], 
            trendline(modelPosErrors[i]), 
            color='red',  
            label='Line of Best Fit'
        )

        # Labels and title
        plt.xlabel("Position Error")
        plt.ylabel("Orientation Error")
        plt.title(f"Position vs. Orientation Errors for {model} Models")
        plt.legend()
        plt.grid(True)

        # save plot
        plt.savefig(f'results/{model}_position_vs_orientation.png')


def plotMSEData(mseValues, models):
    """
    Plot the Mean Squared Error for each model as a bar chart
    
    Args:
        - mseValues (array): List of Mean Squared Errors for each model
        - models (array): List of model names
    """
    plt.figure(figsize=(35, 6))

    # Create bar chart
    plt.bar(models, mseValues)
    
    # Labels and title
    plt.xlabel("Models")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE Comparison of Different Models")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save plot
    plt.savefig("results/mse_data_comparison.png")

    # plot table
    df = pd.DataFrame()
    df['Models'] = models
    df['MSE'] = mseValues
    df.to_csv('results/mse_data_comparison.csv', index=False)


def plotMAEData(maeValues, models):
    """
    Plot the Mean Absolute Error for each model as a bar chart
    
    Args:
        - maeValues (array): List of Mean Absolute Errors for each model
        - models (array): List of model names
    """
    plt.figure(figsize=(35, 6))
        
    # Create bar chart
    plt.bar(models, maeValues)
    
    # Labels and title
    plt.xlabel("Models")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE Comparison of Different Models")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save plot
    plt.savefig("results/mae_data_comparison.png")

    # plot table
    df = pd.DataFrame()
    df['Models'] = models
    df['MAE'] = maeValues
    df.to_csv('results/mae_data_comparison.csv', index=False)



def plotTimings(trainingTimes, testingTimes, models):
    """
    Plot training and testing times for each model as a grouped bar chart

    Args:
        - trainingTimes (array): List of training times for each model
        - testingTimes (array): List of testing times for each model
        - models (array): List of model names
    """
    plt.figure(figsize=(35, 6))
    
    x = np.arange(len(models))  # X-axis positions

    # Create a grouped bar chart
    plt.bar(x - 0.2, trainingTimes, width=0.4, label="Training Time", color='blue')
    plt.bar(x + 0.2, testingTimes, width=0.4, label="Testing Time", color='orange')

    # Labels and title
    plt.xticks(x, models)
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Training vs. Testing Time for Models")
    plt.yscale("log")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Save plot
    plt.savefig("results/timings_comparison.png")

    # plot table
    df = pd.DataFrame()
    df['Models'] = models
    df['Training Times'] = trainingTimes
    df['Testing Times'] = testingTimes
    df.to_csv('results/timings_comparison.csv', index=False)


def plotR2Score(r2Scores, models):
    """
    Plot the R² scores for each model as a bar chart

    Args:
        - r2Scores (array): List of R² scores for each model
        - models (array): List of model names
    """
    plt.figure(figsize=(35, 6))

    # Create bar chart 
    plt.bar(models, r2Scores)
    
    # Labels and title
    plt.xlabel("Models")
    plt.ylabel("R² Score")
    plt.title("R² Scores Comparison of Different Models")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Save plot
    plt.savefig("results/r2_data_comparison.png")

    # plot table
    df = pd.DataFrame()
    df['Models'] = models
    df['R²'] = r2Scores
    df.to_csv('results/r2_data_comparison.csv', index=False)


def storeBestParams(bestParams, models):
    with open('results/best_parameters.txt', 'w') as f:
        for model, param in zip(models, bestParams):
            f.write(f"The model {model}'s best parameters are: {param}")
            f.write("")