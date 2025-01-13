Unsupervised Anomaly Detection
🚀 Project Overview
This project implements unsupervised anomaly detection using machine learning models like KNN, LODA, ABOD, and OCSVM. The primary goal is to analyze cancer-related patient data and identify anomalies for further insights.

🏗️ Project Structure
📂 unsupervised_anomaly_detection
├── 📁 data/                 # Dataset folder
│   └── cancer_issue.csv     # Input dataset
├── 📁 preprocessing/        # Preprocessing scripts
├── 📁 models/               # Model implementations
├── 📁 evaluation/           # Metrics and visualization scripts
├── 📁 scripts/              # Batch processing and report generation
│   ├── batch_processing.py  # Main script for model training and evaluation
│   ├── report_generator.py  # Generates summary report
├── 📁 results/              # Output predictions and reports
├── main.py                  # Main entry point for the project
├── requirements.txt         # Required libraries
└── README.md                # Project documentation

📥 Setup Instructions
Prerequisites
Ensure you have Python 3.8+ installed.

Install Dependencies
Install required Python libraries using:
pip install -r requirements.txt

Run the Project
Open the project folder in Visual Studio Code.
Run the project from the terminal:
python -m main

📊 Features
Data Preprocessing:
    Handles missing values.
    Encodes categorical features.
    Includes PCA for dimensionality reduction.
Models:
    KNN (K-Nearest Neighbors)
    LODA (Lightweight Online Detector of Anomalies)
    ABOD (Angle-Based Outlier Detection)
    OCSVM (One-Class Support Vector Machine)
Evaluation:
    Metrics: Silhouette Score and Dunn Index.
    Summary report generation.
Visualization:
    Visual representations of anomalies for each model.

📝 Output
Example Summary Report
| Model   | Silhouette Score | Dunn Index | Label 0 Instances | Label 1 Instances |
|---------|------------------|------------|-------------------|-------------------|
| ABOD    | 0.41             | 0.05       | 13194             | 166               |
| KNN     | 0.49             | 0.10       | 13226             | 134               |
| LODA    | 0.50             | 0.12       | 13226             | 134               |
| OCSVM   | 0.39             | 0.02       | 13225             | 135               |

Outputs are saved in the results/ folder:
    Model predictions (e.g., KNN_predictions.csv).
    Summary report (summary_report.md).

⚙️ Customization
Modify the following for specific use cases:

Model parameters: Adjust in batch_processing.py or individual model scripts (e.g., models/ocsvm_model.py).
PCA Components: Edit n_components in the PCA function:
apply_pca(X, n_components=10)
