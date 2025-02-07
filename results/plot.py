import matplotlib.pyplot as plt
import numpy as np

def plot_error_distribution(pose_errors):
    errors = np.linalg.norm(pose_errors, axis=1)  # Euclidean norm of errors
    
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Pose Errors')
    plt.xlabel('Error (meters/radians)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
