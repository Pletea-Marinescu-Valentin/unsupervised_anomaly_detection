import os
from scripts.report_generator import generate_summary_report
from scripts.batch_processing import process_dataset

def main():
    process_dataset()
    generate_summary_report("results/batch_processing", "results/summary_report.md")

if __name__ == "__main__":
    main()
