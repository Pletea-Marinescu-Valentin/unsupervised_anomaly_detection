import os
import json
from tabulate import tabulate

def parse_metrics(file_path):
    """
    Parse metrics from a text file and return as a dictionary.
    """
    metrics = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    metrics[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Metrics file not found: {file_path}")
    except Exception as e:
        print(f"Error reading metrics file {file_path}: {e}")
    return metrics

def generate_summary_report(results_dir, output_path="results/summary_report.md"):
    """
    Generate a summary report of evaluation metrics for all models.
    """
    # List all metric files in the results directory
    model_metrics = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_metrics.txt"):  # Match the metrics file pattern
                model_name = file.replace("_metrics.txt", "")
                file_path = os.path.join(root, file)
                try:
                    metrics = parse_metrics(file_path)
                    # Append metrics to the summary table
                    model_metrics.append({
                        "Model": model_name,
                        "Silhouette Score": metrics.get("Silhouette Score", "N/A"),
                        "Dunn Index": metrics.get("Dunn Index", "N/A"),
                        "Label 0 Instances": metrics.get("Label 0", "N/A"),
                        "Label 1 Instances": metrics.get("Label 1", "N/A")
                    })
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    # Write the report to a markdown file
    if model_metrics:
        with open(output_path, "w") as f:
            f.write("# Summary Report\n\n")
            f.write("Evaluation metrics for all models:\n\n")
            f.write(tabulate(model_metrics, headers="keys", tablefmt="github"))
            f.write("\n\n")
        print(f"Summary report saved to {output_path}")
    else:
        print("No metrics found. Ensure that the evaluation metrics are generated.")

if __name__ == "__main__":
    generate_summary_report("results/batch_processing", "results/summary_report.md")
