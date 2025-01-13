from pyod.models.abod import ABOD

def train_abod(X, contamination=0.05):
    """
    Train an ABOD model and return the predictions.
    :param X: Input data (features) as a NumPy array or Pandas DataFrame.
    :return: Predictions (anomaly labels).
    """
    model = ABOD(contamination=contamination)
    model.fit(X)  # Train the model
    y_pred_scores = model.decision_function(X)  # Anomaly scores
    threshold = model.threshold_  # Model-determined threshold for anomalies

    # Convert scores to binary labels: 1 for anomalies, 0 for normal
    y_pred_binary = (y_pred_scores >= threshold).astype(int)
    return y_pred_binary
