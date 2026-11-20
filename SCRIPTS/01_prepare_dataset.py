"""Select the raw GenImage subset and assign splits before image preprocessing.

Run: python SCRIPTS/01_prepare_dataset.py
Requires requests and Pillow. Reads DATA/original/source_inventory.json;
fetches archive indexes and original images, then writes the selected manifest,
source inventory, and preparation audit. Run stages 02 and 03 afterwards.
Selection uses source paths and raw bytes, never the existing final manifest.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps
from remote_zip import DriveRanges, SplitZip

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'DATA' / 'original'
CLEANED = ROOT / 'DATA' / 'cleaned'
SEED = 4002
NAMES={'sd14':'Stable Diffusion 1.4','sd15':'Stable Diffusion 1.5','midjourney':'Midjourney'}
COLUMNS=['image_path','label','label_name','generator','source_collection','original_label','original_split','assigned_split','source_path','source_file_id','source_disk','source_offset','compressed_bytes','zip_crc32','width','height','megapixels','aspect_ratio','format','color_mode','file_bytes','sha256','pixel_sha256','dhash','selection_seed']
# Each tuple: collection, original split, original label, assigned split, target.
PLAN=[('sd14','train','nature','train',1200),('sd14','train','nature','validation',200),
      ('sd14','val','nature','test_seen',200),('sd14','val','nature','test_unseen',600),
      ('sd14','train','ai','train',600),('sd14','train','ai','validation',100),('sd14','val','ai','test_seen',100),
      ('sd15','train','ai','train',600),('sd15','train','ai','validation',100),('sd15','val','ai','test_seen',100),
      ('midjourney','val','ai','test_unseen',600)]

def rank(e):
    """Stable candidate priority within a source pool, independent of RNG versions."""
    return hashlib.sha256(f'{SEED}:{e["source_path"]}'.encode()).hexdigest()

def inspect(data):
    """Decode original bytes and record dimensions, exact hashes, and a visual hash."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        w,h=image.size; fmt=image.format; mode=image.mode
        rgb=ImageOps.exif_transpose(image).convert('RGB')
        pixels=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        small=list(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS).getdata())
        dh=0
        for y in range(8):
            for x in range(8): dh=(dh<<1)|int(small[y*9+x]>small[y*9+x+1])
    return dict(width=w,height=h,megapixels=w*h/1e6,aspect_ratio=w/h,format=fmt,color_mode=mode,file_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),pixel_sha256=pixels,dhash=f'{dh:016x}')

def main(workers=12, cache_dir=None, verify_only=False):
    client=DriveRanges()
    if verify_only:
        # Verification must use cached archive indexes and originals, never downloads.
        def no_network(*args, **kwargs):
            raise RuntimeError('Offline verification requires cached indexes and original images')
        client.get = no_network
    inventory=json.loads((ORIGINAL / 'source_inventory.json').read_text())
    archives={k:SplitZip(k,v,client,cache_dir=cache_dir) for k,v in inventory.items()}
    # Rebuild candidate pools from complete archive indexes, not the saved subset.
    pools={}; original_counts={}
    for k,z in archives.items():
        entries=[e for e in z.index() if Path(e['source_path']).suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}]
        original_counts[k]=dict(Counter('/'.join(e['source_path'].split('/')[1:3]) for e in entries))
        for split in ['train','val']:
            for label in ['ai','nature']:
                pools[k,split,label]=sorted([e for e in entries if e['source_path'].split('/')[1:3]==[split,label]],key=rank)
    if not verify_only:
        (ORIGINAL / 'images').mkdir(parents=True,exist_ok=True)
        CLEANED.mkdir(parents=True,exist_ok=True)
    # Global visited/duplicate state prevents accepted images crossing splits.
    rows=[]; rejected=[]; visited=set(); hashes=set(); pixels=set(); dhashes=[]
    def retrieve(job):
        k,e=job; dest=(ORIGINAL / 'images')/(k+'_'+hashlib.sha256(e['source_path'].encode()).hexdigest()[:24]+Path(e['source_path']).suffix.lower())
        try:
            data=dest.read_bytes() if dest.exists() else archives[k].extract(e)
            # Validate both cached and freshly downloaded bytes against archive metadata.
            import zlib
            if len(data)!=e['uncompressed_bytes'] or zlib.crc32(data)!=e['crc32']: raise ValueError('Cached integrity mismatch')
            meta=inspect(data)
            if not dest.exists(): dest.write_bytes(data)
            return e,dest,meta,None
        except Exception as exc: return e,dest,None,str(exc)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for k,original,label,assigned,target in PLAN:
            candidates=pools[k,original,label]; count=0; pos=0
            # Rejecting a candidate advances to the next ranked path until quota is met.
            while count<target and pos<len(candidates):
                batch=[]
                while len(batch)<min(48,target-count) and pos<len(candidates):
                    e=candidates[pos];pos+=1;key=(k,e['source_path'])
                    if key in visited: continue
                    visited.add(key);batch.append((k,e))
                # executor.map yields in candidate order, regardless of download timing.
                for e,dest,meta,error in executor.map(retrieve,batch):
                    if error and ('Expected partial response' in error or 'Drive download unavailable' in error):
                        raise RuntimeError('Drive access/quota failure; cached images are retained. Restore access before retrying. '+error)
                    reason=error
                    if meta:
                        dh=int(meta['dhash'],16)
                        if meta['sha256'] in hashes or meta['pixel_sha256'] in pixels: reason='exact_duplicate'
                        elif any((dh^old).bit_count()<=4 for old in dhashes): reason='near_duplicate_candidate_dhash_le4'
                    if reason:
                        rejected.append(dict(source_collection=k,source_path=e['source_path'],reason=reason))
                        continue
                    hashes.add(meta['sha256']);pixels.add(meta['pixel_sha256']);dhashes.append(dh)
                    rows.append(dict(image_path=dest.relative_to(ROOT).as_posix(),label=int(label=='ai'),label_name='AI-generated' if label=='ai' else 'Real',generator=NAMES[k] if label=='ai' else 'not_applicable',source_collection=k,original_label=label,original_split=original,assigned_split=assigned,source_path=e['source_path'],source_file_id=archives[k].parts[e['disk']]['id'],source_disk=e['disk'],source_offset=e['offset'],compressed_bytes=e['compressed_bytes'],zip_crc32=f'{e["crc32"]:08x}',selection_seed=SEED,**meta));count+=1
                print(f'{k} {original}/{label} -> {assigned}: {count}/{target}; selected {len(rows)}; transferred {client.bytes_received/1e6:.1f} MB',flush=True)
            if count!=target: raise RuntimeError(f'Insufficient valid distinct images for {k}/{assigned}: {count}/{target}')
    if verify_only:
        # Compare independent selection against the published result, not as input.
        with (CLEANED / 'dataset_manifest.csv').open(newline='', encoding='utf-8') as file:
            published = list(csv.DictReader(file))
        fields = ['source_collection', 'source_path', 'label', 'assigned_split', 'generator']
        actual = [tuple(str(row[field]) for field in fields) + (row['sha256'],) for row in rows]
        expected = [tuple(row[field] for field in fields) + (row.get('original_sha256', row['sha256']),) for row in published]
        if actual != expected:
            raise ValueError('Reconstructed selection, order, labels, splits, or original hashes differ')
        with (CLEANED / 'rejected_candidates.csv').open(newline='', encoding='utf-8') as file:
            if rejected != list(csv.DictReader(file)):
                raise ValueError('Reconstructed rejection log differs')
        print('PASS: independently reconstructed selection, splits, hashes, and rejections match.', flush=True)
        return
    # Publish original measurements and assigned splits as preprocessing input.
    with (CLEANED / 'dataset_manifest.csv').open('w', newline='', encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=COLUMNS);writer.writeheader();writer.writerows(rows)
    with (ORIGINAL / 'source_manifest.csv').open('w', newline='', encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=[c for c in COLUMNS if c != 'assigned_split'],extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    with (CLEANED / 'rejected_candidates.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=['source_collection','source_path','reason']);writer.writeheader();writer.writerows(rejected)
    # Inventory every cached original, including unused candidates from prior runs.
    with (ORIGINAL / 'original_files.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=['filename','file_bytes','sha256']);writer.writeheader()
        for path in sorted((ORIGINAL / 'images').iterdir()):
            if path.is_file():
                raw=path.read_bytes()
                writer.writerow(dict(filename=path.name,file_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    audit=dict(selected=len(rows),class_counts=dict(Counter(r['label_name'] for r in rows)),split_counts=dict(Counter(r['assigned_split'] for r in rows)),rejections=dict(Counter(r['reason'] for r in rejected)),archive_image_counts=original_counts,range_bytes_this_run=client.bytes_received,selected_file_bytes=sum(r['file_bytes'] for r in rows),seed=SEED)
    (CLEANED / 'preparation_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--cache-dir', type=Path, default=ORIGINAL / 'cache')
    parser.add_argument('--verify-only', action='store_true', help='Reconstruct offline and compare without rewriting data')
    args=parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    main(args.workers, args.cache_dir, args.verify_only)
