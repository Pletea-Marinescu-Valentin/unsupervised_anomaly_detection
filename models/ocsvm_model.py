from sklearn.svm import OneClassSVM
import numpy as np

def train_ocsvm(X, contamination=0.01):
    """
    Train an OCSVM model for anomaly detection with manually set parameters.
    :param X: Input data (features) as a NumPy array or Pandas DataFrame.
    :param contamination: Approximate fraction of outliers in the dataset.
    :return: Predictions (anomaly labels).
    """
    print(f"Training OCSVM with contamination={contamination}, gamma=0.01")
    
    # Manually set parameters for the OCSVM model
    nu = contamination  # Proportion of outliers
    gamma = 0.01        # Gamma for the RBF kernel
    
    # Initialize and fit the One-Class SVM
    model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
    model.fit(X)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Convert predictions to binary: 0 for normal, 1 for anomalies
    y_pred_binary = (y_pred == -1).astype(int)
    
    print(f"OCSVM training completed. Predictions: {np.unique(y_pred_binary, return_counts=True)}")
    return y_pred_binary
