import pandas as pd

def save_predictions_to_csv(X, predictions, filename):
    """Save predictions to a CSV file."""
    results = pd.DataFrame(X)
    results["Predictions"] = predictions
    results.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")
