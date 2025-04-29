import pandas as pd

def save_predictions_to_csv(X, predictions, filename):
    """Save predictions to a CSV file."""
    results = pd.DataFrame(X)
    results["Predictions"] = predictions
    results.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def load_predictions_from_csv(filename):
    """Load predictions from a CSV file."""
    try:
        data = pd.read_csv(filename)
        print(f"Predictions loaded from {filename}")
        return data
    except Exception as e:
        print(f"Error loading predictions from {filename}: {e}")
        return None
