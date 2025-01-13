from pyod.models.loda import LODA

def train_loda(X, contamination=0.1):
    """
    Train a LODA model and return the predictions.
    :param X: Input data (features) as a NumPy array or Pandas DataFrame.
    :param contamination: The proportion of outliers in the data (default: 0.002).
    :return: Predictions (anomaly labels).
    """
    model = LODA(contamination=contamination)
    model.fit(X)  # Train the model
    y_pred_scores = model.decision_function(X)  # Anomaly scores
    threshold = model.threshold_  # Model-determined threshold for anomalies

    # Convert scores to binary labels: 1 for anomalies, 0 for normal
    y_pred_binary = (y_pred_scores >= threshold).astype(int)
    return y_pred_binary
