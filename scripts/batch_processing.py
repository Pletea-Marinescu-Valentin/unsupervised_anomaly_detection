import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from preprocessing.data_loading import load_dataset
from preprocessing.data_cleaning import remove_missing_values
from preprocessing.data_scaling import scale_data
from models.knn_model import train_knn
from models.ocsvm_model import train_ocsvm
from models.abod_model import train_abod
from models.loda_model import train_loda
from evaluation.visualization import visualize_anomalies
from evaluation.metrics import evaluate_model

def feature_engineering(df):
    """
    Enhance dataset with derived features and remove irrelevant ones.
    """
    # Remove irrelevant features
    if 'PatientID' in df.columns:
        df = df.drop(columns=['PatientID'])
    
    # Add derived features
    df['BMI_TumorSize'] = df['BMI'] * df['TumorSize']
    df['Age_SurvivalRatio'] = df['Age'] / (df['SurvivalMonths'] + 1)  # Avoid division by zero
    
    return df

def apply_pca(X, n_components=10):
    """
    Reduce dimensionality using PCA.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    print(f"PCA completed. Explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")
    return X_reduced

def encode_categorical_columns(df):
    """
    Encode categorical columns to numeric values.
    """
    # Binary encoding for 'Yes'/'No' columns
    binary_columns = ['FamilyHistory', 'Recurrence']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Encode SmokingStatus as numeric
    smoking_mapping = {'Non-Smoker': 0, 'Former Smoker': 1, 'Smoker': 2}
    df['SmokingStatus'] = df['SmokingStatus'].map(smoking_mapping)

    # One-hot encode GeneticMarker
    df = pd.get_dummies(df, columns=['GeneticMarker'], drop_first=True)

    # One-hot encoding for other categorical columns
    categorical_columns = ['Gender', 'Race/Ethnicity', 'CancerType', 'Stage', 
                           'TreatmentType', 'TreatmentResponse', 'HospitalRegion']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    return df

def process_dataset():
    """
    Process the dataset for unsupervised anomaly detection.
    """
    # Step 1: Load dataset
    dataset_path = "data/cancer_issue.csv"
    print(f"Processing dataset: {dataset_path}")
    df = load_dataset(dataset_path)
    print("Dataset loaded successfully.")
    print(df.head())

    # Step 2: Remove missing values
    df = remove_missing_values(df)
    print(f"Removed missing values. Remaining rows: {len(df)}")

    # Step 3: Feature Engineering
    df = feature_engineering(df)
    print("Feature engineering completed.")

    # Step 4: Encode categorical columns
    df = encode_categorical_columns(df)
    print("Categorical columns encoded.")

    # Step 5: Select numeric columns for scaling
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    X = scale_data(df, numeric_columns)
    print("Data scaled successfully.")

    # Step 6: Apply PCA
    X = apply_pca(X, n_components=10)

    # Step 7: Prepare results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Step 8: Train and evaluate models
    models = {
        "KNN": train_knn,
        "OCSVM": train_ocsvm,
        "ABOD": train_abod,
        "LODA": train_loda
    }

    predictions = {}
    for model_name, train_function in models.items():
        print(f"Training {model_name} model...")
        y_pred = train_function(X, contamination=0.01)  # Adjust contamination
        print(f"Predictions: {y_pred}, Length: {len(y_pred)}")  # Debugging
        predictions[model_name] = y_pred

        # Save predictions
        output_path = os.path.join(results_dir, f"{model_name}_predictions.csv")
        pd.DataFrame({'Prediction': y_pred}).to_csv(output_path, index=False)
        print(f"{model_name} predictions saved to {output_path}")

        # Evaluate model
        evaluate_model(X, y_pred, model_name)

        # Visualize anomalies
        visualize_anomalies(df, model_name, y_pred)

if __name__ == "__main__":
    process_dataset()
