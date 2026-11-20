"""
PURPOSE:
Reproduces the data-loading and model-initialization stage of the analysis.
Does NOT train or evaluate on the held-out test sets.

Run from the project root:
    python SCRIPTS/05_prepare_data_and_model.py

Requires: torch, torchvision, pandas, Pillow.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


class AIImageDataset(Dataset):
    """Read cleaned JPEGs and binary labels from one manifest split."""

    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row['resolved_path']) as image:
            tensor = self.transform(image.convert('RGB'))
        label = torch.tensor(float(row['label']), dtype=torch.float32)
        return tensor, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'DATA/cleaned/dataset_manifest.csv')
    parser.add_argument('--images', type=Path, default=Path(__file__).resolve().parents[1] / 'DATA/cleaned/images')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if not args.manifest.is_file():
        parser.error(f'Manifest not found: {args.manifest}')
    if not args.images.is_dir():
        parser.error(f'Images directory not found: {args.images}')

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest = pd.read_csv(args.manifest)
    required = {'image_path', 'label', 'assigned_split'}
    if not required.issubset(manifest.columns):
        parser.error(f'Missing manifest columns: {sorted(required - set(manifest.columns))}')

    # Resolve the repository-relative manifest paths against extracted JPEGs.
    # Reject duplicate basenames to avoid silently matching the wrong image.
    physical_images = list(args.images.rglob('*.jpg')) + list(args.images.rglob('*.jpeg'))
    lookup = {}
    for path in physical_images:
        if path.name in lookup:
            parser.error(f'Duplicate JPEG basename: {path.name}')
        lookup[path.name] = str(path)
    manifest['resolved_path'] = manifest['image_path'].map(
        lambda p: lookup.get(Path(str(p)).name)
    )
    if manifest['resolved_path'].isna().any():
        examples = manifest.loc[manifest['resolved_path'].isna(), 'image_path'].head(5).tolist()
        parser.error(f'Missing images referenced in manifest: {examples}')
    if manifest['resolved_path'].duplicated().any():
        parser.error('Multiple manifest rows map to the same physical image')

    expected = {'train': 2400, 'validation': 400, 'test_seen': 400, 'test_unseen': 1200}
    actual = manifest['assigned_split'].value_counts().to_dict()
    if actual != expected:
        parser.error(f'Unexpected split counts: {actual}')
    if set(pd.to_numeric(manifest['label'], errors='coerce').dropna().unique()) != {0, 1}:
        parser.error('Expected binary labels 0 (real) and 1 (AI-generated)')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    loaders = {}
    for split in expected:
        frame = manifest.loc[manifest['assigned_split'] == split].copy()
        loaders[split] = DataLoader(
            AIImageDataset(frame, transform),
            batch_size=args.batch_size,
            shuffle=(split == 'train'),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        print(f'{split}: {len(frame)} images')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    images, labels = next(iter(loaders['train']))
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
    if outputs.shape != (len(labels), 1):
        raise RuntimeError(f'Unexpected output shape: {tuple(outputs.shape)}')

    print('Input shape:', tuple(images.shape))
    print('Output shape:', tuple(outputs.shape))
    print('Final layer:', model.fc)
    print('Loss:', type(criterion).__name__)
    print('Optimizer:', type(optimizer).__name__)
    print('PASS: DataLoaders and ResNet-50 forward pass are ready for training.')
    print('Note: No training or held-out test evaluation was performed.')


if __name__ == '__main__':
    main()
