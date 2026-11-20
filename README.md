# DS4002_Project3

The original and cleaned dataset images are included as regular Git files. Clone the repository to download them:

```text
git clone https://github.com/msoly25/DS4002_Project3.git
```

For an existing clone, run `git pull` in the project folder. No Git LFS setup is needed. The images total approximately 1.91 GiB (4,707 originals and 4,400 cleaned images), with additional disk space needed for Git history.

See [DATA/README.md](DATA/README.md) for dataset details and [download instructions](DATA/original/GET_DATA.md) for source restoration. Use `DATA/cleaned/dataset_manifest.csv` to select the 4,400 modeling images.

## Data preprocessing

To recreate the cleaned modeling images from the included originals, run these commands from the project root before validation and modeling:

```text
python -m pip install -r DATA/cleaned/requirements.txt
python SCRIPTS/00_preprocess.py
```

The script preserves the originals and assigned splits, and writes the cleaned images, updated manifest, and preprocessing audit to `DATA/cleaned/`.
