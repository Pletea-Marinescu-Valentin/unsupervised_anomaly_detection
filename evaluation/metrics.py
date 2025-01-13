import numpy as np
from sklearn.metrics import silhouette_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import pairwise_distances

def dunn_index(X, labels):
    """
    Compute the Dunn Index for cluster separation.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    # Compute intra-cluster distances
    intra_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_distances.append(np.max(pairwise_distances(cluster_points)))

    # Compute inter-cluster distances
    inter_distances = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1:]:
            cluster_points1 = X[labels == label1]
            cluster_points2 = X[labels == label2]
            if len(cluster_points1) > 0 and len(cluster_points2) > 0:
                inter_distances.append(np.min(pairwise_distances(cluster_points1, cluster_points2)))

    if not inter_distances or not intra_distances:
        return np.nan

    return np.min(inter_distances) / np.max(intra_distances)

def evaluate_model(X, y_pred, model_name):
    """
    Evaluate the performance of a given model.
    """
    print(f"Evaluating model: {model_name}")

    # Silhouette Score
    try:
        silhouette = silhouette_score(X, y_pred)
        print(f"Silhouette Score: {silhouette:.2f}")
    except Exception as e:
        print(f"Error computing Silhouette Score: {e}")
        silhouette = np.nan

    # Dunn Index
    try:
        dunn = dunn_index(X, y_pred)
        if np.isnan(dunn):
            print("Dunn Index: Not computable (insufficient data)")
        else:
            print(f"Dunn Index: {dunn:.2f}")
    except Exception as e:
        print(f"Error computing Dunn Index: {e}")
        dunn = np.nan

    # Generate a basic summary
    print(f"Anomaly Prediction Summary for {model_name}:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count} instances")

    # Save metrics to a results file
    metrics_file = f"results/batch_processing/{model_name}_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Silhouette Score: {silhouette:.2f}\n")
        f.write(f"Dunn Index: {dunn:.2f}\n")
        for label, count in zip(unique, counts):
            f.write(f"Label {label}: {count} instances\n")
    print(f"Metrics saved to {metrics_file}")
