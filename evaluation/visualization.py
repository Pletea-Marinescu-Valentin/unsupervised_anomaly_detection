import matplotlib.pyplot as plt
import seaborn as sns

def visualize_anomalies(df, model_name, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df['BMI'],
        y=df['TumorSize'],
        hue=y_pred,
        palette={0: "blue", 1: "red"},
        legend="full"
    )
    plt.title(f"Anomalies detected by {model_name}")
    plt.xlabel("BMI")
    plt.ylabel("Tumor Size")
    plt.show()
