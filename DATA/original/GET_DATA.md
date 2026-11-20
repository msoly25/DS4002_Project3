# Obtain the original images

Both image folders are included as regular Git files: 4,707 unchanged original images (approximately 1.77 GiB) and 4,400 cleaned modeling images (approximately 137 MiB).

## Download from this repository

```text
git clone https://github.com/msoly25/DS4002_Project3.git
```

Open the cloned folder in VS Code to view the images. No Git LFS setup is needed. The full collection contains approximately 1.91 GiB of image data, plus additional disk space for Git history. For an existing clone, run `git pull` to retrieve the committed images.

The original-source instructions below are an alternative for reconstructing the analyzed subset. The source links were recorded during acquisition on November 12, 2026; availability was not rechecked during this update.

## Reproduce selection from raw sources

From the project root, install `DATA/cleaned/requirements.txt` and run `python SCRIPTS/01_prepare_dataset.py`. The script reads `DATA/original/source_inventory.json` and the original archive indexes. It ranks source paths by SHA-256 of `4002:<source_path>` within each collection/source-split/class pool, then fills the split quotas in fixed plan order. It rejects unreadable images, duplicate file/pixel hashes, and difference hashes within distance 4 of any previously selected image. The next ranked candidate replaces each rejection. All Midjourney images are assigned to the unseen test set.

The script reuses locally cached originals after verifying archive size and CRC32; otherwise it downloads the required ranges. It writes the original-image manifest, source manifest, local inventory, rejection log, and preparation audit. Run `python SCRIPTS/02_explore_data.py` and then `python SCRIPTS/03_preprocess.py` to finish exploratory validation and create the final modeling dataset. Archive caches are generated at runtime under `DATA/original/cache/` and need not be committed. Download interruptions stop the run rather than silently changing source collections.

## Restore the exact subset manually

1. Open the official collection links below. Download all multipart ZIP segments for the relevant collection into one directory, then extract with a multipart-ZIP-capable tool such as 7-Zip. These archives are much larger than the subset. Do not treat each segment as an independent ZIP.
2. For each row in `../cleaned/dataset_manifest.csv`, locate `source_path` within the extracted `source_collection`. Copy that file, without re-encoding it, to `original_image_path` relative to the project root. The manifest supplies the exact destination filename, so no new random sample is needed.
3. Verify its file length against `original_file_bytes` and SHA-256 against `original_sha256`. Keep the final manifest's labels and assigned splits unchanged. Only these 4,400 records belong to the analyzed dataset.
4. The original local download pool also contains 307 unused candidates. `original_files.csv` preserves names, sizes, and hashes for all 4,707 local files; source mappings for every unused candidate were not retained, so the instructions above reproduce the analyzed subset, not every unused download.

The final dataset now contains standardized modeling copies in `../cleaned/images/`. After restoring originals, run `python -m pip install -r DATA/cleaned/requirements.txt` and `python SCRIPTS/03_preprocess.py` from the project root. This applies EXIF orientation, RGB conversion, aspect-ratio-preserving resizing of the shorter side to 256 pixels, a 224 x 224 center crop, and uniform JPEG encoding at quality 95 with 4:4:4 chroma subsampling. The originals remain unchanged. `image_path` in the final manifest points to the processed copy; `original_image_path` and `original_*` measurements identify the source. Join `source_manifest.csv` on its `image_path` to the final manifest's `original_image_path` to recover split assignments. This manual restoration uses the existing manifest to recover the selected bytes. The numbered source-preparation workflow above independently regenerates selection and splits from the raw archive indexes.

## Original sources and acquisition history


Start at the [official repository](https://github.com/GenImage-Dataset/GenImage), then follow its [Google Drive dataset link](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS). The root contains ADM, BigGAN, GLIDE, Midjourney, Stable Diffusion 1.4, Stable Diffusion 1.5, VQDM, and Wukong collections. This project uses:

| Collection | Authors' Drive folder |
|---|---|
| Stable Diffusion 1.4 | [stable_diffusion_v_1_4](https://drive.google.com/drive/folders/12xighYOtu-ryfYEUnNrSeZqrxT8P08Zy) |
| Stable Diffusion 1.5 | [stable_diffusion_v_1_5](https://drive.google.com/drive/folders/1lG4WheCh3_CM2XNrRwfXVeOiG_1gZ_K2) |
| Midjourney | [Midjourney](https://drive.google.com/drive/folders/1GLZedAqYuBh0kIQCZcbY_RspPJsR7ZyE) |

Inside these folders, files are multipart ZIP segments (`.z01`, `.z02`, ..., `.zip`), not separately downloadable images or independent small ZIPs. Most segments are 3 GiB. The final `.zip` is not a standalone subset. The observed unique filename sets total approximately 96.4 GB (SD1.4), 98.5 GB (SD1.5), and 227.5 GB (Midjourney); these totals describe archives, not local storage requirements. SD1.5 has two IDs for the same `.z26` filename; the original acquisition chose the lexicographically first ID consistently.

**Observed access limit:** An initial 4,800-image plan reached Google Drive download quotas during the final Midjourney requests (the response was HTML titled `Google Drive - Quota exceeded`). The final plan was reduced to 4,400 images, still within the requested range, using 600 Midjourney AI images and 600 real controls in the unseen test. All final selections were reconstructed from already cached original images. Earlier acquisition transferred at least 1.885 GB of image ranges plus about 0.135 GB of indexes; the final cached preparation run required no network. Unused downloaded candidates are retained in the local original image pool.

**Verified small-download method:** Google's public download endpoint accepted HTTP `Range` requests with status 206 and matching `Content-Range`. The original acquisition procedure read each ZIP central directory, located sampled image entries across split segments, and requested only their compressed bytes plus a small header allowance. It checked decompression, byte length, and CRC32. It refused responses that would download a full archive. No Hugging Face or Kaggle data are used. Google Drive's archive preview is not relied on.

The repository also links an authors' Baidu mirror (access code `ztf1`); this run did not use or test that mirror. Public Drive access worked without account credentials for this run, but quotas, removed files, or changed range support could prevent future downloads. In that case obtain an authors-provided smaller export or restore access rather than silently substituting another dataset.

