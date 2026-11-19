"""
Purpose: JPEG recompression sensitivity analysis.
Uses a frozen, previously trained ResNet-50 checkpoint; does not retrain.
Evaluates original standardized images and JPEG recompression at qualities 75/50
on the held-out test_seen and test_unseen splits.

Outputs (in jpeg_sensitivity_outputs/):
    jpeg_sensitivity_metrics.csv
    jpeg_sensitivity_predictions.csv
    jpeg_sensitivity_comparison.csv
    jpeg_sensitivity_by_generator.csv
    jpeg_accuracy_comparison.png
    jpeg_f1_comparison.png
    jpeg_sensitivity_summary.json

Run in Colab: !python /content/05_jpeg_sensitivity.py
Or locally: python 05_jpeg_sensitivity.py --base-dir /path/to/DSProject3_MI3
"""

# Read optional command-line settings so the same script works in Colab or locally.
import argparse
# BytesIO allows JPEG recompression in memory without altering source images.
import io
import json
# Extract images.zip only when the required test images are not already present.
import zipfile
from pathlib import Path

import matplotlib
# Use a non-interactive backend: figures are saved to disk, not displayed.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Dataset wrapper that applies the SAME preprocessing for every test condition.
# The only intended difference is whether an image is recompressed.
class JPEGTestDataset(Dataset):
    def __init__(self, dataframe, quality=None):
        # Reset indices so DataLoader positions map directly to the correct rows.
        self.df = dataframe.reset_index(drop=True)
        self.quality = quality
        # Normalize with ImageNet statistics expected by the pretrained ResNet-50.
        # Do not use training augmentation during evaluation.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # Always start with the exact same standardized source image.
        with Image.open(row["resolved_path"]) as source:
            image = source.convert("RGB")
        # None means the unchanged original; numeric qualities create JPEG copies.
        # Recompression is in memory and never overwrites dataset files.
        if self.quality is not None:
            with io.BytesIO() as buffer:
                # Keep chroma subsampling fixed so quality is the tested setting.
                image.save(buffer, format="JPEG", quality=self.quality, subsampling=0)
                buffer.seek(0)
                with Image.open(buffer) as compressed:
                    image = compressed.convert("RGB")
        # Binary ground truth: 0 = real photograph, 1 = AI-generated.
        return self.transform(image), int(row["label"])


# Match manifest entries to actual files, regardless of ZIP folder nesting.
def resolve_images(manifest, extract_dir, zip_path):
    # The manifest may contain repository-relative paths, so match by filename.
    # Reject duplicate filenames to prevent silently assigning the wrong image.
    def build_lookup():
        paths = list(extract_dir.rglob("*.jpg")) + list(extract_dir.rglob("*.jpeg"))
        lookup = {}
        for path in paths:
            if path.name in lookup and lookup[path.name] != str(path):
                raise ValueError(f"Duplicate image basename: {path.name}")
            lookup[path.name] = str(path)
        return lookup

    lookup = build_lookup()
    missing = [p for p in manifest["image_path"] if Path(str(p)).name not in lookup]
    # Only extract when needed; avoid repeating potentially expensive ZIP work.
    if missing and zip_path.is_file():
        print(f"Extracting {zip_path} to {extract_dir} ...", flush=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            # Guard against ZIP entries attempting to write outside extract_dir.
            for member in archive.infolist():
                target = (extract_dir / member.filename).resolve()
                if not target.is_relative_to(extract_dir.resolve()):
                    raise ValueError("Unsafe path inside images.zip")
            archive.extractall(extract_dir)
        lookup = build_lookup()
    # Preserve the original manifest and add a resolved absolute path column.
    manifest = manifest.copy()
    manifest["resolved_path"] = manifest["image_path"].map(
        lambda p: lookup.get(Path(str(p)).name)
    )
    return manifest
            probabilities.extend(scores.cpu().tolist())
            # Fixed 0.5 decision threshold: >= 0.5 predicts AI (class 1).
            predicted.extend((scores >= 0.5).long().cpu().tolist())
            labels.extend(batch_labels.tolist())
    name = "Original" if quality is None else f"JPEG {quality}"
    # Compare predictions against true labels; zero_division avoids undefined
    # precision/recall when a condition predicts no positive examples.
    metrics = {
        "split": str(frame["assigned_split"].iloc[0]),
        "condition": name,
        "jpeg_quality": "original" if quality is None else quality,
        "n_images": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1_score": f1_score(labels, predicted, zero_division=0),
    }
    # Save one row per image so individual errors can be inspected later.
    result = frame[["image_path", "assigned_split", "generator", "label"]].reset_index(drop=True).copy()
    result["condition"] = name
    result["predicted_label"] = predicted
    result["ai_probability"] = probabilities
    result["correct"] = result["label"] == result["predicted_label"]
    return metrics, result


# Coordinate input validation, model loading, experiment runs, and outputs.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="/content/drive/MyDrive/DSProject3_MI3")
    parser.add_argument("--extract-dir", default="/content/mi3_data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    # Colab may supply unrelated kernel flags; ignore those safely.
    args, _ = parser.parse_known_args()  # Compatible with notebook-injected arguments.

    base = Path(args.base_dir)
    output = base / "jpeg_sensitivity_outputs"
    # Keep experiment outputs separate from training and baseline evaluation.
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = base / "dataset_manifest.csv"
    checkpoint_path = base / "model_outputs" / "resnet50_best_model.pth"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Expected {manifest_path} and {checkpoint_path}. Check --base-dir.")

    manifest = pd.read_csv(manifest_path)
    # These columns identify each image, its class, generator, and held-out split.
    required = {"image_path", "label", "generator", "assigned_split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Missing manifest columns: {required - set(manifest.columns)}")
    # Never use training or validation images in this final robustness test.
    manifest = manifest[manifest["assigned_split"].isin(["test_seen", "test_unseen"])].copy()
    manifest = resolve_images(manifest, Path(args.extract_dir), base / "images.zip")
    if manifest["resolved_path"].isna().any():
        missing = manifest.loc[manifest["resolved_path"].isna(), "image_path"].head().tolist()
        raise FileNotFoundError(f"Test images not found after extraction: {missing}")
    if len(manifest) != 1600:
        raise ValueError(f"Expected 1600 held-out test images, found {len(manifest)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 # Reconstruct the same one-logit classification architecture used in training.
    # weights=None avoids downloading new weights; the checkpoint supplies them.
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    # Restore the previously selected checkpoint without retraining or tuning.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device).eval()
    print(f"Device: {device}; test images: {len(manifest)}", flush=True)

    metrics_rows, prediction_frames = [], []
    # Keep familiar and unfamiliar generator results separate.
    for split in ("test_seen", "test_unseen"):
        frame = manifest[manifest["assigned_split"] == split].copy()
        # Repeat on identical test images: original, moderate, strong compression.
        for quality in (None, 75, 50):
            condition = "Original" if quality is None else f"JPEG {quality}"
            print(f"Evaluating {split} | {condition} ...", flush=True)
            metrics, predictions = evaluate(model, device, frame, quality,
                                            args.batch_size, args.num_workers)
            metrics_rows.append(metrics)
            prediction_frames.append(predictions)
            print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}", flush=True)
    # Export both aggregate metrics and per-image predictions as CSV tables.
    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_df.to_csv(output / "jpeg_sensitivity_metrics.csv", index=False)
    predictions_df.to_csv(output / "jpeg_sensitivity_predictions.csv", index=False)

    # Use each split's original performance as its own reference point.
    baseline = metrics_df.loc[metrics_df["condition"] == "Original",
                              ["split", "accuracy", "recall", "f1_score"]].rename(columns={
        "accuracy": "baseline_accuracy", "recall": "baseline_recall", "f1_score": "baseline_f1"
    })
    comparison = metrics_df.merge(baseline, on="split", how="left")
    for metric, baseline_name in (("accuracy", "baseline_accuracy"),
                                  ("recall", "baseline_recall"),
                                  ("f1_score", "baseline_f1")):
        # Multiply by 100 to express the difference in percentage points.
        # Negative values indicate worse performance after recompression.
        comparison[f"{metric}_change_pp"] = 100 * (comparison[metric] - comparison[baseline_name])
    comparison.to_csv(output / "jpeg_sensitivity_comparison.csv", index=False)

    # AI recall answers: of the truly AI-generated images, what share
    # did the model correctly label as AI? Group by generator and condition.
    ai_only = predictions_df[predictions_df["label"] == 1]
    by_generator = (ai_only.groupby(["assigned_split", "condition", "generator"], as_index=False)
                    .agg(n_ai_images=("label", "size"), ai_recall=("predicted_label", "mean")))
    by_generator.to_csv(output / "jpeg_sensitivity_by_generator.csv", index=False)

    # Plot the same ordered conditions for both splits to compare trends.
    condition_order = ["Original", "JPEG 75", "JPEG 50"]
    for metric, title, filename in (
        ("accuracy", "Accuracy", "jpeg_accuracy_comparison.png"),
        ("f1_score", "F1-score", "jpeg_f1_comparison.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for split in ("test_seen", "test_unseen"):
            subset = metrics_df[metrics_df["split"] == split].set_index("condition")
            ax.plot(condition_order, subset.loc[condition_order, metric].values,
                    marker="o", linewidth=2, label=split)
        ax.set(title=f"ResNet-50 JPEG Sensitivity: {title}",
               xlabel="JPEG condition", ylabel=title, ylim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        # Save a high-resolution figure suitable for the GitHub OUTPUT folder.
        fig.savefig(output / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

# Machine-readable record of the settings and results for reproducibility.
    summary = {
        "model": "ResNet-50 (saved checkpoint; no retraining)",
        "checkpoint": str(checkpoint_path),
        "experiment": "JPEG recompression sensitivity",
        "jpeg_conditions": condition_order,
        "jpeg_recompression_subsampling": 0,
        "classification_threshold": 0.5,
        "test_images": len(manifest),
        "image_instances_evaluated": len(predictions_df),
        "results": json.loads(metrics_df.to_json(orient="records")),
    }
    with (output / "jpeg_sensitivity_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    print("\nResults:\n", metrics_df.to_string(index=False))
    print(f"\nSaved outputs to: {output}")
    for path in sorted(output.iterdir()):
        print(" -", path.name)


# Run only when invoked as a script, not when imported into another module.
if __name__ == "__main__":
    main()
