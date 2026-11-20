"""
Step 4: Physical File Validation

Purpose:
Verify the integrity and organization of the 4,400 cleaned
images before training the ResNet-50 classification model.

Checks:
1. Dataset contains 4,400 records and physical JPEG files.
2. All manifest image paths resolve to existing files.
3. Images are readable, RGB, JPEG, and 224 x 224 pixels.
4. Images have no duplicate file paths or identical file bytes.
5. File SHA-256 hashes match the dataset manifest.
6. Training, validation, and test splits are correct.
7. Real and AI-generated labels are balanced.
8. Midjourney AI images appear only in test_unseen.

Output:
A JSON validation report containing PASS/FAIL results.

Required packages:
pandas, Pillow
"""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd
from PIL import Image


# 1. FILE LOCATIONS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / 'DATA' / 'cleaned' / 'dataset_manifest.csv'
IMAGE_DIR = ROOT / 'DATA' / 'cleaned' / 'images'
OUTPUT_PATH = ROOT / 'OUTPUT' / 'validation_report.json'

EXPECTED_SPLITS = {
    "train": 2400,
    "validation": 400,
    "test_seen": 400,
    "test_unseen": 1200
}

EXPECTED_LABELS = {
    0: 2200,
    1: 2200
}

EXPECTED_SPLIT_LABELS = {
    "train": {0: 1200, 1: 1200},
    "validation": {0: 200, 1: 200},
    "test_seen": {0: 200, 1: 200},
    "test_unseen": {0: 600, 1: 600}
}

# 2. HELPER FUNCTIONS
# --------------------------------------------------

def calculate_sha256(path):
    """Calculate a file's SHA-256 hash."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():

    print("DS 4002 Project 3")
    print("STEP 4: PHYSICAL FILE VALIDATION")
    print("-" * 45)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    results = {}
    details = {}

    def check(name, passed, detail=None):
        results[name] = bool(passed)
        details[name] = detail
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")

  
    # 3. MANIFEST VALIDATION
    # --------------------------------------------------

    required_columns = {
        "image_path",
        "label",
        "assigned_split",
        "generator"
    }

    check(
        "Required manifest columns",
        required_columns.issubset(manifest.columns)
    )

    if not required_columns.issubset(manifest.columns):
        raise ValueError("Missing required manifest columns.")

    check(
        "Manifest contains 4400 records",
        len(manifest) == 4400,
        {"actual": len(manifest)}
    )

    check(
        "Required metadata is complete",
        not manifest[list(required_columns)].isna().any().any()
    )

    check(
        "Manifest image paths are unique",
        manifest["image_path"].is_unique
    )

    # Resolve manifest entries to the repository cleaned-image directory.
    manifest["resolved_path"] = manifest[
        "image_path"
    ].apply(
        lambda p: IMAGE_DIR / Path(str(p)).name
    )

    check(
        "Resolved image paths are unique",
        manifest["resolved_path"].is_unique
    )

  
    # 4. LABEL AND SPLIT VALIDATION
    # --------------------------------------------------

    split_counts = manifest[
        "assigned_split"
    ].value_counts().to_dict()

    check(
        "Expected dataset splits",
        split_counts == EXPECTED_SPLITS,
        split_counts
    )

    numeric_labels = pd.to_numeric(
        manifest["label"], errors="coerce"
    )

    label_counts = {
        int(k): int(v)
        for k, v in numeric_labels.value_counts().items()
        if pd.notna(k)
    }

    check(
        "Balanced overall labels",
        label_counts == EXPECTED_LABELS,
        label_counts
    )

    split_label_counts = {}

    for split in EXPECTED_SPLITS:
        subset = manifest[
            manifest["assigned_split"] == split
        ]

        counts = pd.to_numeric(
            subset["label"], errors="coerce"
        ).value_counts()

        split_label_counts[split] = {
            0: int(counts.get(0, 0)),
            1: int(counts.get(1, 0))
        }

    check(
        "Balanced labels within splits",
        split_label_counts == EXPECTED_SPLIT_LABELS,
        split_label_counts
    )

    # Midjourney AI images must be withheld from training.
    ai_rows = manifest[numeric_labels == 1]

    midjourney_rows = ai_rows[
        ai_rows["generator"].astype(str).str.contains(
            "midjourney", case=False, na=False
        )
    ]

    midjourney_valid = (
        len(midjourney_rows) == 600
        and midjourney_rows[
            "assigned_split"
        ].eq("test_unseen").all()
    )

    check(
        "Midjourney held out for unseen testing",
        midjourney_valid,
        {
            "midjourney_ai_images": len(midjourney_rows),
            "splits": midjourney_rows[
                "assigned_split"
            ].value_counts().to_dict()
        }
    )

    # 5. PHYSICAL IMAGE VALIDATION
    # --------------------------------------------------

    missing = []
    unreadable = []
    wrong_size = []
    wrong_format = []
    wrong_mode = []
    hash_mismatches = []

    actual_hashes = {}
    found_paths = set()

    for _, row in manifest.iterrows():

        path = Path(row["resolved_path"])

        if not path.is_file():
            missing.append(str(path))
            continue

        try:
            with Image.open(path) as img:
                size = img.size
                image_format = img.format
                color_mode = img.mode
                img.verify()

            if size != (224, 224):
                wrong_size.append(str(path))

            if image_format != "JPEG":
                wrong_format.append(str(path))

            if color_mode != "RGB":
                wrong_mode.append(str(path))

            actual_hash = calculate_sha256(path)

            actual_hashes.setdefault(
                actual_hash, []
            ).append(str(path))

            if (
                "sha256" in manifest.columns
                and pd.notna(row["sha256"])
                and str(row["sha256"]).lower() != actual_hash
            ):
                hash_mismatches.append(str(path))

            found_paths.add(path.resolve())

        except Exception as error:
            unreadable.append({
                "path": str(path),
                "error": str(error)
            })

    check(
        "All manifest images exist",
        len(missing) == 0,
        {"count": len(missing), "examples": missing[:5]}
    )

    check(
        "All images are readable",
        len(unreadable) == 0,
        {"count": len(unreadable), "examples": unreadable[:5]}
    )

    check(
        "All images are 224 x 224",
        len(wrong_size) == 0,
        {"count": len(wrong_size), "examples": wrong_size[:5]}
    )

    check(
        "All images use JPEG format",
        len(wrong_format) == 0,
        {"count": len(wrong_format), "examples": wrong_format[:5]}
    )

    check(
        "All images use RGB color",
        len(wrong_mode) == 0,
        {"count": len(wrong_mode), "examples": wrong_mode[:5]}
    )

    # Detect duplicate file contents.
    duplicate_groups = [
        paths
        for paths in actual_hashes.values()
        if len(paths) > 1
    ]

    check(
        "No duplicate image file contents",
        len(duplicate_groups) == 0,
        {
            "duplicate_groups": len(duplicate_groups),
            "examples": duplicate_groups[:5]
        }
    )

    if "sha256" in manifest.columns:
        check(
            "File hashes match manifest",
            len(hash_mismatches) == 0,
            {
                "mismatches": len(hash_mismatches),
                "examples": hash_mismatches[:5]
            }
        )

    # Count physical JPEG files.
    physical_jpegs = {
        path.resolve()
        for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg"}
    }

    check(
        "Exactly 4400 physical JPEG files",
        len(physical_jpegs) == 4400,
        {"actual": len(physical_jpegs)}
    )

    extra_files = physical_jpegs - found_paths

    check(
        "No unlisted JPEG files",
        len(extra_files) == 0,
        {
            "count": len(extra_files),
            "examples": [
                str(p) for p in list(extra_files)[:5]
            ]
        }
    )

  
    # 6. SAVE VALIDATION REPORT
    # --------------------------------------------------

    overall_pass = all(results.values())

    report = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "manifest_path": str(MANIFEST_PATH),
        "image_directory": str(IMAGE_DIR),
        "total_manifest_records": len(manifest),
        "total_physical_jpegs": len(physical_jpegs),
        "checks": results,
        "details": details,
        "overall_result": (
            "PASS" if overall_pass else "FAIL"
        )
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(OUTPUT_PATH, "w") as file:
        json.dump(report, file, indent=2)

    print("-" * 45)
    print(
        "OVERALL RESULT:",
        report["overall_result"]
    )
    print("Validation report saved to:")
    print(OUTPUT_PATH)

    if not overall_pass:
        raise RuntimeError(
            "One or more validation checks failed. "
            "Review the results above."
        )


if __name__ == "__main__":
    main()
