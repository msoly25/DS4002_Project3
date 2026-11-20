"""

PURPOSE:
Evaluate an already-trained ResNet-50 binary classifier on two held-out test
splits. No model parameters are updated and no training is performed here.

Labels: 0 = real photograph; 1 = AI-generated image.

Outputs in OUTPUT/:
  test_metrics.csv                 - one row of metrics per test split
  test_predictions.csv             - individual image predictions/probabilities
  confusion_matrix_test_seen.png  - classification counts for familiar generators
  confusion_matrix_test_unseen.png - classification counts for unfamiliar generator
  generator_ai_recall.csv          - AI detection recall by generator
  generator_ai_recall.png          - bar chart of generator-specific recall
  evaluation_summary.json         - settings and key results in structured form

Run from the project root:
  python SCRIPTS/07_evaluate_resnet50.py
Use --base-dir to select another clone of this repository.
"""

# Standard-library modules handle command-line settings, structured JSON,
# ZIP extraction, and portable file/folder paths.
import argparse
import json
import zipfile
from pathlib import Path

# Use a noninteractive plotting backend so figures can be saved when the script
# runs in Colab or from a terminal without opening a separate plot window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pandas handles the image manifest and results tables; PyTorch loads the model
# and runs inference; scikit-learn computes classification performance metrics.
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class TestImages(Dataset):
    """Read held-out images and apply the same normalization used in training."""

    def __init__(self, dataframe):
        # Reset row numbers so DataLoader indices match this filtered split.
        self.frame = dataframe.reset_index(drop=True)

        # ResNet-50 expects tensors normalized using ImageNet mean and standard
        # deviation. Do not introduce new augmentation during evaluation.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        # DataLoader uses this to know how many test images are available.
        return len(self.frame)

    def __getitem__(self, index):
        # Locate one image from the resolved path recorded in the manifest.
        row = self.frame.iloc[index]
        with Image.open(row["resolved_path"]) as opened:
            # RGB ensures every input has the three channels expected by ResNet.
            image = opened.convert("RGB")

        # Return the normalized image and its ground-truth class label.
        return self.transform(image), int(row["label"])


def resolve_images(frame, extract_dir, zip_path):
    """Match manifest filenames to extracted files; extract ZIP if necessary."""

    def lookup_images():
        # Manifest paths may contain repository-relative prefixes, so identify
        # each actual image by its filename inside the extracted directory.
        lookup = {}
        for path in list(extract_dir.rglob("*.jpg")) + list(extract_dir.rglob("*.jpeg")):
            # Duplicate filenames would make the mapping ambiguous and could
            # accidentally associate a manifest row with the wrong image.
            if path.name in lookup and lookup[path.name] != str(path):
                raise ValueError(f"Duplicate image basename: {path.name}")
            lookup[path.name] = str(path)
        return lookup

    lookup = lookup_images()
# If any test image is not extracted yet, use the original images ZIP.
    if any(Path(str(p)).name not in lookup for p in frame["image_path"]):
        if not zip_path.is_file():
            raise FileNotFoundError(f"Missing extracted images and ZIP: {zip_path}")
        print(f"Extracting images from {zip_path} ...", flush=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as archive:
            # Reject ZIP entries that try to write outside the extraction folder.
            for item in archive.infolist():
                target = (extract_dir / item.filename).resolve()
                if not target.is_relative_to(extract_dir.resolve()):
                    raise ValueError("Unsafe path in images.zip")
            archive.extractall(extract_dir)

        # Refresh file paths after extraction so all test images can be found.
        lookup = lookup_images()

    frame = frame.copy()
    frame["resolved_path"] = frame["image_path"].map(
        lambda path: lookup.get(Path(str(path)).name)
    )

    # Stop rather than silently evaluating an incomplete test set.
    if frame["resolved_path"].isna().any():
        missing = frame.loc[frame["resolved_path"].isna(), "image_path"].head().tolist()
        raise FileNotFoundError(f"Images not found: {missing}")
    return frame


def evaluate_split(model, device, frame, batch_size, num_workers):
    """Predict labels and probabilities, then compute metrics for one split."""

    # shuffle=False preserves manifest order, allowing predictions to be
    # correctly paired with each image's original metadata in the saved CSV.
    loader = DataLoader(TestImages(frame), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    labels, predictions, probabilities = [], [], []

    # Evaluation mode disables training-specific behavior such as dropout and
    # updates to batch-normalization statistics.
    model.eval()

    # Inference mode reduces memory usage because no gradients are needed.
    with torch.inference_mode():
        for images, targets in loader:
            # The final model layer outputs logits; sigmoid converts them into
            # probabilities between 0 and 1 for the AI-generated class.
            probs = torch.sigmoid(model(images.to(device))).flatten().cpu()
            labels.extend(targets.tolist())
            probabilities.extend(probs.tolist())

            # A fixed 0.5 threshold maps probabilities to binary predictions:
            # probability >= 0.5 means AI (1), otherwise real (0).
            predictions.extend((probs >= 0.5).long().tolist())

    split = str(frame["assigned_split"].iloc[0])

    # Accuracy: fraction of all images classified correctly.
    # Precision: fraction of predicted AI images that actually are AI.
    # Recall: fraction of actual AI images correctly identified as AI.
    # F1-score: harmonic mean of precision and recall.
    # zero_division=0 avoids an error if a metric has no positive predictions.
    metrics = {
        "split": split,
        "n_images": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1_score": f1_score(labels, predictions, zero_division=0),
    }

    # Save individual predictions alongside image metadata for later analysis.
    # resolved_path is omitted because it points to a temporary/local location.
    results = frame.drop(columns=["resolved_path"]).reset_index(drop=True).copy()
    results["true_label"] = labels
    results["predicted_label"] = predictions
    results["ai_probability"] = probabilities
    results["correct"] = results["true_label"] == results["predicted_label"]
    return metrics, results


def save_confusion_matrix(frame, split, output_dir):
    """Save a 2-by-2 chart showing correct classifications and error types."""

    # Rows represent actual labels; columns represent predicted labels.
    # [0,0] = correctly identified real; [0,1] = real mistaken for AI.
    # [1,0] = AI mistaken for real; [1,1] = correctly identified AI.
    cm = confusion_matrix(frame["true_label"], frame["predicted_label"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Real (0)", "AI (1)"])
    ax.set_yticks([0, 1], labels=["Real (0)", "AI (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix: {split}")

    # Print each cell's count inside the heatmap for easier interpretation.
    # Switch text color on darker cells to keep the numbers readable.
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                    color="white" if cm[row, col] > cm.max() / 2 else "black",
                    fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{split}.png", dpi=300)
    plt.close(fig)  # Free figure memory after saving.
    return cm.tolist()  # Convert NumPy values for JSON serialization.


def main():
    """Load inputs, evaluate both held-out splits, and write output files."""

    # Default locations use this repository, including when cloned in Colab.
    # Command-line flags allow someone else to reproduce the analysis locally.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args, _ = parser.parse_known_args()  # Ignore Colab notebook-injected arguments.

    # Keep output files separate from the manifest and saved model checkpoint.
    base = Path(args.base_dir)
    output_dir = base / "OUTPUT"
    manifest_path = base / "DATA" / "cleaned" / "dataset_manifest.csv"
    checkpoint_path = base / "OUTPUT" / "resnet50_best_model.pth"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Missing dataset_manifest.csv or model checkpoint under " + str(base))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Confirm the manifest has the columns needed to identify and group images.
    manifest = pd.read_csv(manifest_path)
    required = {"image_path", "label", "generator", "assigned_split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Missing manifest columns: {required - set(manifest.columns)}")

    # Exclude train/validation entirely: this script evaluates held-out data only.
    manifest = manifest[manifest["assigned_split"].isin(["test_seen", "test_unseen"])].copy()
    if len(manifest) != 1600:
        raise ValueError(f"Expected 1600 held-out images, found {len(manifest)}")
    manifest = resolve_images(manifest, (Path(args.extract_dir) if args.extract_dir else base / "DATA" / "cleaned" / "images"), base / "DATA" / "cleaned" / "images.zip")

    # Load the same ResNet-50 architecture used during training, including its
    # one-output binary head. weights=None avoids downloading new weights:
    # the trained parameters come from the saved .pth checkpoint instead.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device).eval()
    print(f"Loaded checkpoint; device: {device}", flush=True)

    # Evaluate the familiar-generator and unfamiliar-generator splits separately
    # so their performance can be compared rather than averaged together.
    metrics_rows, prediction_frames, confusion_matrices = [], [], {}
    for split in ["test_seen", "test_unseen"]:
        frame = manifest[manifest["assigned_split"] == split].copy()
        metrics, predictions = evaluate_split(model, device, frame,
                                              args.batch_size, args.num_workers)
        metrics_rows.append(metrics)
        prediction_frames.append(predictions)
        confusion_matrices[split] = save_confusion_matrix(predictions, split, output_dir)
        print(f"{split}: accuracy={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1_score']:.4f}", flush=True)

    # Save both summary-level metrics and image-level predictions as CSV tables.
    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Generator-specific recall only considers true AI images (label 1).
    # Because predictions are binary 0/1, their mean within each AI-only group
    # equals the fraction correctly predicted as AI: the group's AI recall.
    ai_only = predictions_df[predictions_df["true_label"] == 1]
    generator_df = (ai_only.groupby(["assigned_split", "generator"], dropna=False)
                    .agg(n_ai_images=("true_label", "size"),
                         ai_recall=("predicted_label", "mean"))
                    .reset_index().rename(columns={"assigned_split": "split"}))
    generator_df.to_csv(output_dir / "generator_ai_recall.csv", index=False)

    # Create a visual comparison of AI recall for each generator and test split.
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r.generator}\n({r.split})" for r in generator_df.itertuples()]
    bars = ax.bar(labels, generator_df["ai_recall"] * 100)
    ax.set_ylim(0, 100)
    ax.set_ylabel("AI detection recall (%)")
    ax.set_title("AI Recall by Generator")


# Annotate each bar with the exact percentage for readability.
    for bar, value in zip(bars, generator_df["ai_recall"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{value:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "generator_ai_recall.png", dpi=300)
    plt.close(fig)

    # The JSON file records the checkpoint, decision threshold, test metrics,
    # confusion matrices, and generator-level results in one reproducible record.
    summary = {
        "model": "ResNet-50",
        "checkpoint": str(checkpoint_path),
        "classification_threshold": 0.5,
        "metrics": metrics_rows,
        "confusion_matrices": confusion_matrices,
        "generator_ai_recall": generator_df.to_dict(orient="records"),
    }
    with open(output_dir / "evaluation_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    # Show key results in the terminal in addition to writing them to OUTPUT/.
    print("\nSaved evaluation outputs to:", output_dir)
    print(metrics_df.to_string(index=False))
    print("\nGenerator-specific AI recall:")
    print(generator_df.to_string(index=False))


if __name__ == "__main__":
    # This guard runs main() when the file is executed as a script, but not when
    # its functions/classes are imported into another Python file.
    main()
  
