"""Build and verify modeling copies offline; originals and splits never change.

Run from the project root: python SCRIPTS/00_preprocess.py
Requires Pillow==13.0.0 (see DATA/cleaned/requirements.txt).
The manifest is published only after all output checks pass. Reruns always
read originals, never previously compressed modeling copies.
"""
import csv
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps, JpegImagePlugin, __version__ as pillow_version

# Anchor data locations to the repository, regardless of the working directory.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'DATA'
CLEANED_DIR = DATA_DIR / 'cleaned'
NATIVE = ['image_path', 'width', 'height', 'megapixels', 'aspect_ratio',
          'format', 'color_mode', 'file_bytes', 'sha256', 'pixel_sha256', 'dhash']
EXPECTED = {
    ('train', '0'): 1200, ('train', '1'): 1200,
    ('validation', '0'): 200, ('validation', '1'): 200,
    ('test_seen', '0'): 200, ('test_seen', '1'): 200,
    ('test_unseen', '0'): 600, ('test_unseen', '1'): 600,
}
SETTINGS = dict(exif_orientation='applied before resize', color_mode='RGB',
                resize_short_side=256, resize_rounding='floor long side',
                resampling='LANCZOS', crop='center; floor left/top offsets',
                output_size=[224, 224], format='JPEG', quality=95,
                subsampling='4:4:4 (Pillow subsampling=0)',
                optimize=False, progressive=False,
                metadata='no source EXIF, ICC profile, or comments copied')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def resolved(relative):
    relative = Path(relative)
    # Existing manifests use lowercase data/. Resolve that legacy spelling to
    # DATA/ as well on case-sensitive systems, without rewriting provenance.
    if relative.parts and relative.parts[0].lower() == 'data':
        relative = Path('DATA', *relative.parts[1:])
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path outside project: {relative}')
    return path


def metadata(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        w, h = image.size
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        small = list(rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS).getdata())
        dh = 0
        for y in range(8):
            for x in range(8):
                dh = (dh << 1) | int(small[y*9+x] > small[y*9+x+1])
        return dict(width=w, height=h, megapixels=w*h/1e6, aspect_ratio=w/h,
                    format=image.format, color_mode=image.mode, file_bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=pixels,
                    dhash=f'{dh:016x}')


def convert(row):
    row = dict(row)
    for name in NATIVE:
        row.setdefault('original_' + name, row[name])
    source = resolved(row['original_image_path'])
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['original_sha256']:
        raise ValueError(f'Original hash mismatch: {source}')
    with Image.open(io.BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        w, h = rgb.size
        size = (256, max(256, h*256//w)) if w <= h else (max(256, w*256//h), 256)
        resized = rgb.resize(size, Image.Resampling.LANCZOS)
        left, top = (size[0]-224)//2, (size[1]-224)//2
        crop = resized.crop((left, top, left+224, top+224))
        # A fresh image strips inherited metadata and makes encoder settings uniform.
        clean = Image.frombytes('RGB', (224, 224), crop.tobytes())
        encoded = io.BytesIO()
        clean.save(encoded, format='JPEG', quality=95, subsampling=0,
                   optimize=False, progressive=False)
    output = CLEANED_DIR / 'images' / (source.stem + '.jpg')
    raw = encoded.getvalue()
    if not output.exists() or output.read_bytes() != raw:
        temporary = output.with_suffix('.tmp')
        temporary.write_bytes(raw)
        temporary.replace(output)
    # Inspect bytes read back from disk, not just the in-memory encoding.
    saved = output.read_bytes()
    meta = metadata(saved)
    with Image.open(io.BytesIO(saved)) as check:
        if (check.format, check.mode, check.size) != ('JPEG', 'RGB', (224, 224)):
            raise ValueError(f'Incorrect output format: {output}')
        if JpegImagePlugin.get_sampling(check) != 0 or check.getexif():
            raise ValueError(f'Incorrect subsampling or residual EXIF: {output}')
        tables = json.dumps(check.quantization, sort_keys=True)
    row.update(meta)
    row['image_path'] = output.relative_to(ROOT).as_posix()
    return row, tables


def main():
    manifest = CLEANED_DIR / 'dataset_manifest.csv'
    rows = read_csv(manifest)
    before = [(r['source_collection'], r['source_path'], r['label'], r['assigned_split']) for r in rows]
    if Counter((r['assigned_split'], r['label']) for r in rows) != EXPECTED:
        raise ValueError('Split counts do not match the frozen plan')
    mj = [r for r in rows if r['generator'] == 'Midjourney']
    if len(mj) != 600 or any(r['assigned_split'] != 'test_unseen' for r in mj):
        raise ValueError('Midjourney holdout violated')
    if any(r['label'] == '1' and r['generator'] != 'Midjourney'
           for r in rows if r['assigned_split'] == 'test_unseen'):
        raise ValueError('Unexpected unseen generator')
    names = [Path(r.get('original_image_path', r['image_path'])).stem for r in rows]
    if len(set(names)) != 4400:
        raise ValueError('Output filename collision')
    (CLEANED_DIR / 'images').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = []
        for i, result in enumerate(executor.map(convert, rows), 1):
            outputs.append(result)
            if i % 400 == 0:
                print(f'Encoded and verified {i}/4400 images', flush=True)
    final = [r for r, _ in outputs]
    if len({tables for _, tables in outputs}) != 1:
        raise ValueError('JPEG quantization tables differ')
    for key in ['image_path', 'sha256', 'pixel_sha256', 'original_sha256', 'original_pixel_sha256']:
        if len({r[key] for r in final}) != 4400:
            raise ValueError(f'Duplicate {key}; manifest not published')
    if before != [(r['source_collection'], r['source_path'], r['label'], r['assigned_split']) for r in final]:
        raise ValueError('Source, label or split changed')
    dh = [int(r['dhash'], 16) for r in final]
    near = [(i, j) for i in range(len(dh)) for j in range(i+1, len(dh)) if (dh[i]^dh[j]).bit_count() <= 4]
    # Preserve every original, including the 307 unused downloads.
    inventory = read_csv(DATA_DIR / 'original' / 'original_files.csv')
    for r in inventory:
        path = DATA_DIR / 'original' / 'images' / r['filename']
        if hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError(f'Original file changed: {path}')
    if near:
        raise ValueError(f'{len(near)} near-duplicate output pairs require review; manifest not published')
    audit = dict(total_images=len(final), settings=SETTINGS,
                 pillow_version=pillow_version, jpeg_library=Image.core.jpeglib_version,
                 output_format_counts=dict(Counter(r['format'] for r in final)),
                 output_dimensions='224x224', output_mode='RGB',
                 output_bytes=sum(r['file_bytes'] for r in final),
                 original_files_verified=len(inventory), originals_unchanged=True,
                 labels_and_splits_unchanged=True, midjourney_only_in_test_unseen=True,
                 duplicate_file_hashes=0, duplicate_pixel_hashes=0,
                 near_duplicate_pairs_dhash_le4=len(near),
                 uniform_jpeg_quantization_tables=True,
                 split_counts={s: {'real': EXPECTED[s, '0'], 'ai': EXPECTED[s, '1']}
                               for s in ['train', 'validation', 'test_seen', 'test_unseen']},
                 normalization='Not baked into JPEGs. Apply pretrained weights normalization when loading tensors.',
                 limitations=['Center cropping removes peripheral content.',
                              'Re-encoding does not erase all prior compression artifacts.',
                              'Perceptual hashing cannot exclude every transformed duplicate.'])
    temp = manifest.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(final[0]))
        writer.writeheader(); writer.writerows(final)
    temp.replace(manifest)
    (CLEANED_DIR / 'preprocessing_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
