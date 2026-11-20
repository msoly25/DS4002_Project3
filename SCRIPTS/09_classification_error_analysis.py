
"""
Step 9: Classification Error Analysis

Purpose:
Reproduce false-positive counts, false-negative counts,
false-positive rates, and Wilson 95% confidence intervals
using existing test metrics and dataset split counts.

Inputs:
OUTPUT/test_metrics.csv
DATA/cleaned/split_counts.csv

Output:
OUTPUT/classification_error_analysis.csv

Run:
python SCRIPTS/09_classification_error_analysis.py

No trained model checkpoint is required.
"""

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "OUTPUT/test_metrics.csv"
COUNTS = ROOT / "DATA/cleaned/split_counts.csv"
OUTPUT = ROOT / "OUTPUT/classification_error_analysis.csv"

# Load previously generated test metrics.
with METRICS.open(newline="") as file:
    metrics = list(csv.DictReader(file))

# Load the number of real and AI images in each split.
with COUNTS.open(newline="") as file:
    split_counts = {
        row["assigned_split"]: row
        for row in csv.DictReader(file)
    }

results = []

for row in metrics:
    split = row["split"]
    n = int(row["n_images"])

    counts = split_counts[split]
    real_count = int(counts["Real"])
    ai_count = n - real_count

    precision = float(row["precision"])
    recall = float(row["recall"])

    # Reconstruct the integer confusion-matrix counts.
    true_positive = round(recall * ai_count)
    false_negative = ai_count - true_positive

    if precision <= 0:
        raise ValueError("Cannot reconstruct FP from zero precision")

    false_positive = round(true_positive / precision - true_positive)

    # Check consistency with the recorded accuracy.
    correct = n - false_positive - false_negative
    accuracy = correct / n

    if abs(accuracy - float(row["accuracy"])) > 1e-8:
        raise ValueError(f"Inconsistent metrics for {split}")

    false_positive_rate = false_positive / real_count

    # Wilson score 95% confidence interval.
    z = 1.96
    denominator = 1 + z**2 / n

    center = (
        accuracy + z**2 / (2 * n)
    ) / denominator

    margin = (
        z * math.sqrt(
            accuracy * (1 - accuracy) / n
            + z**2 / (4 * n**2)
        )
    ) / denominator

    results.append({
        "split": split,
        "n_images": n,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "false_positive_rate": f"{false_positive_rate:.4f}",
        "accuracy": f"{accuracy:.4f}",
        "accuracy_ci_lower_95": f"{center - margin:.4f}",
        "accuracy_ci_upper_95": f"{center + margin:.4f}",
    })

# Save the results in the OUTPUT directory.
with OUTPUT.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Classification error analysis complete.")
print(f"Saved: {OUTPUT}")
