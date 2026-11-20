"""

PURPOSE: Loads the validated 4,400-image manifest, preserves the assigned train and
validation splits, freezes ImageNet-pretrained ResNet-50 features, and trains
its binary classification head. Selects the best checkpoint by validation
loss (early stopping), then writes training curves, history, and summary.

Default paths use DATA/cleaned/ and OUTPUT/ in this repository.
Run from the project root:
    python SCRIPTS/06_train_resnet50.py

Requirements: torch, torchvision, pandas, Pillow, numpy, matplotlib.
The test_seen and test_unseen splits are NEVER used in training or selection.
"""

# Allow type annotations to be interpreted consistently across Python versions.
from __future__ import annotations

# Standard libraries for command-line settings, reproducibility, and file output.
import argparse
import copy
import json
import random
from pathlib import Path

# Plotting and data-processing libraries used for training diagnostics.
import matplotlib
# Save plots without requiring an interactive display (useful for Colab scripts).
matplotlib.use("Agg")
# Plotting and data-processing libraries used for training diagnostics.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# PyTorch and torchvision provide the dataset pipeline and pretrained model.
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


# Expected counts protect against accidentally training on the held-out test sets.
EXPECTED_SPLITS = {"train": 2400, "validation": 400, "test_seen": 400, "test_unseen": 1200}


# Wrap each manifest row as an image tensor paired with its binary label.
class AIImageDataset(Dataset):
    """Manifest-backed images with pretrained ImageNet normalization."""

    def __init__(self, dataframe: pd.DataFrame):
        # Reset row positions so DataLoader indexing stays predictable.
        self.dataframe = dataframe.reset_index(drop=True)
        # Match the normalization expected by ImageNet-pretrained ResNet-50.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # DataLoader uses this to determine how many samples are available.
    def __len__(self) -> int:
        return len(self.dataframe)

    # Load one image and return its normalized tensor and numeric class label.
    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        # Open the file only while needed and convert it to three RGB channels.
        with Image.open(row["resolved_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label


# Verify the manifest and image files before beginning a potentially long run.
def load_manifest(manifest_path: Path, images_dir: Path) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    df = pd.read_csv(manifest_path)
    # The model requires image paths, class labels, and predefined split names.
    required = {"image_path", "label", "assigned_split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(df.columns))}")
    # The manifest stores repository-style paths; images are located by filename.
    df["resolved_path"] = df["image_path"].map(lambda value: str(images_dir / Path(str(value)).name))
    # Fail early if any image listed in the manifest cannot be found.
    missing = df.loc[~df["resolved_path"].map(lambda value: Path(value).is_file())]
    if not missing.empty:
        raise FileNotFoundError(f"{len(missing)} manifest images missing; example: {missing.iloc[0]['resolved_path']}")
    if df["resolved_path"].duplicated().any():
        raise ValueError("Duplicate resolved image paths detected")
    # Preserve the original train/validation/test partition sizes.
    if df['assigned_split'].value_counts().to_dict() != EXPECTED_SPLITS:
        raise ValueError('Unexpected split counts')
    if not df['label'].isin([0, 1]).all():
        raise ValueError('Expected binary labels')
    return df


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one pass, updating only the classification head during training."""
    # Keep frozen feature extraction and its BatchNorm statistics fixed.
    model.eval()
    model.fc.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images).flatten()
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += ((logits >= 0) == labels.bool()).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=root / 'DATA/cleaned/dataset_manifest.csv')
    parser.add_argument('--images', type=Path, default=root / 'DATA/cleaned/images')
    parser.add_argument('--output', type=Path, default=root / 'OUTPUT')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.patience) < 1 or args.num_workers < 0 or args.learning_rate <= 0:
        parser.error('Positive batch size, epochs, patience, learning rate and nonnegative workers required')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    df = load_manifest(args.manifest, args.images)
    train_df = df.loc[df['assigned_split'] == 'train'].copy()
    val_df = df.loc[df['assigned_split'] == 'validation'].copy()
    train_loader = DataLoader(AIImageDataset(train_df), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(AIImageDataset(val_df), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # Freeze the backbone: do not fine-tune convolutional layers.
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Replace the 1000-class head with one output for binary classification.
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    # Numerically stable binary cross-entropy for raw logits.
    criterion = nn.BCEWithLogitsLoss()
    # Train the new classification head using Adam.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "resnet50_best_model.pth"
    # Record metrics each epoch for later tables and plots.
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    # Select the checkpoint with the lowest validation loss.
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    # Train, validate, and consider saving the model after each epoch.
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        for name, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[name].append(float(value))
        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}",
            flush=True,
        )
        # Save a checkpoint only when validation loss improves.
        if val_loss < best_val_loss:
            # Select the checkpoint with the lowest validation loss.
            best_val_loss = val_loss
            best_epoch = epoch
            # Copy weights so subsequent training cannot change the saved best state.
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_path)
            epochs_without_improvement = 0
            print("  Best model saved.", flush=True)
        else:
            epochs_without_improvement += 1
            # Stop after the configured number of epochs without improvement.
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.", flush=True)
                break

    if best_model_state is None:
        raise RuntimeError("Training finished without a valid checkpoint")
    # Restore the best validation-loss checkpoint, not necessarily the last epoch.
    model.load_state_dict(best_model_state)
    model.eval()

    # Save numerical training history as a CSV for reproducibility.
    history_df = pd.DataFrame(history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_path = args.output / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    # Visualize both loss and accuracy for training vs. validation.
    epochs = history_df["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    axes[0].set(title="ResNet-50: Training and Validation Loss", xlabel="Epoch", ylabel="Binary Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Training Accuracy")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
    axes[1].set(title="ResNet-50: Training and Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    plot_path = args.output / "training_validation_curves.png"
    # Write a high-resolution PNG suitable for GitHub reports.
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Capture the key hyperparameters, split sizes, and selected checkpoint.
    summary = {
        "model": "ResNet-50",
        "pretrained_weights": "ImageNet (ResNet50_Weights.DEFAULT)",
        "training_images": len(train_df),
        "validation_images": len(val_df),
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "loss_function": "BCEWithLogitsLoss",
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_val_loss),
        "best_validation_accuracy": history["val_accuracy"][best_epoch - 1],
        "random_seed": args.seed,
        "device": str(device),
        "model_path": str(model_path),
        "test_sets_used_for_training_or_selection": False,
    }
    summary_path = args.output / "training_summary.json"
    # Store machine-readable metadata alongside the model and plots.
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Report where all generated files were saved.
    print("\nTraining complete!", flush=True)
    print(f"Best epoch: {best_epoch} | Best validation loss: {best_val_loss:.4f}", flush=True)
    for path in (model_path, plot_path, history_path, summary_path):
        print(f"Saved: {path}", flush=True)


if __name__ == '__main__':
    main()
