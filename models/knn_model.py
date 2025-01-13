from pyod.models.knn import KNN

def train_knn(X, contamination=0.1):
    """
    Train a KNN model and return the predictions.
    :param X: Input data (features) as a NumPy array or Pandas DataFrame.
    :return: Predictions (anomaly scores or labels).
    """
    model = KNN(n_neighbors=10, contamination=contamination)
    model.fit(X)  # Train the model on the data
    y_pred = model.labels_  # Binary labels (0 for normal, 1 for anomaly)
    return y_pred
