"""Validate selected raw images and recreate the exploratory tables and plots.

Run: python SCRIPTS/02_explore_data.py after stage 01.
Reads DATA/cleaned/dataset_manifest.csv and original image bytes; writes EDA
and quality audit files to DATA/cleaned/. Also works after stage 03 by using
original_* measurements. Requires pandas, NumPy, matplotlib, Pillow, requests.
"""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / 'DATA' / 'cleaned'
# Numbered scripts are loaded by path rather than Python module-name syntax.
spec = importlib.util.spec_from_file_location('prepare_dataset', Path(__file__).with_name('01_prepare_dataset.py'))
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)
inspect = preparation.inspect

GROUPS=['Real','Stable Diffusion 1.4','Stable Diffusion 1.5','Midjourney']
COLORS=['#467599','#22A699','#E7A33E','#9867B8']
SPLIT_COUNTS={
    'train': [1200,600,600,0],
    'validation': [200,100,100,0],
    'test_seen': [200,100,100,0],
    'test_unseen': [600,0,0,600],
}

def binary_metrics(labels, predictions):
    """AI is positive; undefined precision/recall/F1 use zero_division=0."""
    y=np.asarray(labels); p=np.asarray(predictions)
    if y.ndim!=1 or y.shape!=p.shape or not len(y):
        raise ValueError('Expected equally sized, nonempty binary vectors')
    if not np.isin(y,[0,1]).all() or not np.isin(p,[0,1]).all():
        raise ValueError('Labels and predictions must be 0 (real) or 1 (AI)')
    tp=int(((y==1)&(p==1)).sum()); fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum())
    return dict(accuracy=float((y==p).mean()),
                precision=tp/(tp+fp) if tp+fp else 0.0,
                recall=tp/(tp+fn) if tp+fn else 0.0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)

def validate_splits(df):
    counts=pd.crosstab(df.assigned_split,df.group)
    expected=pd.DataFrame.from_dict(SPLIT_COUNTS,orient='index',columns=GROUPS)
    if set(counts.index)!=set(expected.index) or set(counts.columns)!=set(GROUPS):
        raise ValueError('Unexpected split or generator groups')
    if not counts.reindex(index=expected.index,columns=GROUPS).eq(expected).all().all():
        raise ValueError('Split counts differ from the frozen 4,400-image plan')
    checks={}
    for split in ['test_seen','test_unseen']:
        y=df.loc[df.assigned_split.eq(split),'label'].to_numpy()
        if set(y)!={0,1}:
            raise ValueError(f'{split} requires both real and AI images')
        # Synthetic predictions test metric plumbing, not model performance.
        assert binary_metrics(y,y)==dict(accuracy=1.0,precision=1.0,recall=1.0,f1=1.0)
        assert binary_metrics(y,1-y)==dict(accuracy=0.0,precision=0.0,recall=0.0,f1=0.0)
        assert binary_metrics(y,np.zeros_like(y))==dict(accuracy=.5,precision=0.0,recall=0.0,f1=0.0)
        assert binary_metrics(y,np.ones_like(y))==dict(accuracy=.5,precision=.5,recall=1.0,f1=2/3)
        checks[split]=dict(real=int((y==0).sum()),ai=int((y==1).sum()),
                           supported_metrics=['accuracy','precision','recall','f1'],
                           synthetic_prediction_checks='passed; not model scores')
    return checks

def load_manifest():
    df=pd.read_csv(CLEANED / 'dataset_manifest.csv',dtype={'zip_crc32':str,'dhash':str,'original_dhash':str})
    # EDA always describes raw originals, even when rerun after preprocessing.
    for name in ['image_path', 'width', 'height', 'megapixels', 'aspect_ratio',
                 'format', 'color_mode', 'file_bytes', 'sha256', 'pixel_sha256', 'dhash']:
        if 'original_' + name in df.columns:
            df[name] = df['original_' + name]
    df = df[preparation.COLUMNS].copy()
    df['group']=df['generator'].where(df['label'].eq(1),'Real')
    return df

def validate(df):
    metric_checks=validate_splits(df)
    missing=[]; unreadable=[]; changed=[]
    for row in df.itertuples():
        relative=Path(row.image_path)
        if relative.parts[0].lower() == 'data':
            relative=Path('DATA', *relative.parts[1:])
        path=ROOT / relative
        if not path.is_file(): missing.append(row.image_path);continue
        try:
            meta=inspect(path.read_bytes())
            if any(str(meta[c])!=str(getattr(row,c)) for c in ['width','height','sha256','pixel_sha256','dhash','file_bytes']): changed.append(row.image_path)
        except Exception as exc: unreadable.append([row.image_path,str(exc)])
    dh=[int(x,16) for x in df.dhash]
    near_pairs=sum((a^b).bit_count()<=4 for i,a in enumerate(dh) for b in dh[i+1:])
    result=dict(total_images=len(df),class_counts=df.label_name.value_counts().to_dict(),missing_images=missing,unreadable_images=unreadable,metadata_mismatches=changed,duplicate_paths=int(df.image_path.duplicated().sum()),duplicate_file_hashes=int(df.sha256.duplicated().sum()),duplicate_pixel_hashes=int(df.pixel_sha256.duplicated().sum()),near_duplicate_pairs_dhash_le4=near_pairs,missing_metadata_cells=int(df.isna().sum().sum()))
    assert len(df)==4400 and df.label.value_counts().to_dict()=={0:2200,1:2200}
    assert not missing and not unreadable and not changed
    assert not any(result[k] for k in ['duplicate_paths','duplicate_file_hashes','duplicate_pixel_hashes','near_duplicate_pairs_dhash_le4','missing_metadata_cells'])
    assert df.loc[df.generator.eq('Midjourney'),'assigned_split'].eq('test_unseen').all()
    assert not df.loc[df.assigned_split.isin(['train','validation']),'generator'].eq('Midjourney').any()
    assert df.loc[df.assigned_split.isin(['train','validation']),'original_split'].eq('train').all()
    assert df.loc[df.assigned_split.str.startswith('test'),'original_split'].eq('val').all()
    assert df.groupby(['source_collection','source_path']).size().max()==1
    assert df.groupby('assigned_split').label.mean().eq(.5).all()
    result['test_metric_readiness']=metric_checks
    (CLEANED / 'quality_audit.json').write_text(json.dumps(result,indent=2))
    return result

def plots(df):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    class_counts=df.label_name.value_counts().reindex(['Real','AI-generated'])
    bars=axs[0].bar(class_counts.index,class_counts.values,color=[COLORS[0],COLORS[1]],width=.6)
    axs[0].bar_label(bars,fmt='{:,.0f}',padding=5);axs[0].set(title='Class balance',ylabel='Number of images',xlabel='Original class',ylim=(0,2800))
    counts=df.loc[df.label.eq(1),'generator'].value_counts().reindex(GROUPS[1:])
    bars=axs[1].bar(['SD 1.4','SD 1.5','Midjourney\n(held out)'],counts.values,color=COLORS[1:],width=.6)
    axs[1].bar_label(bars,fmt='{:,.0f}',padding=5);axs[1].set(title='AI images by generator',ylabel='Number of images',xlabel='Generator',ylim=(0,950))
    fig.suptitle(f'GenImage subset composition | {len(df):,} images',fontsize=16,fontweight='bold')
    fig.savefig(CLEANED / 'plot1.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    distributions=[df.loc[df.group.eq(group),'megapixels'].to_numpy() for group in GROUPS]
    boxes=axs[0].boxplot(distributions,orientation='horizontal',tick_labels=[f'{g}\n(n={len(x):,})' for g,x in zip(GROUPS,distributions)],patch_artist=True,widths=.5,flierprops={'markersize':3,'alpha':.35})
    for box,color in zip(boxes['boxes'],COLORS): box.set_facecolor(color);box.set_alpha(.65)
    for i,(x,color) in enumerate(zip(distributions,COLORS),1): axs[0].scatter(np.median(x),i,color=color,s=45,zorder=5)
    for group,color in zip(GROUPS,COLORS):
        sub=df[df.group.eq(group)]
        axs[1].scatter(sub.width,sub.height,s=12,alpha=.35 if group=='Real' else .9,color=color,label=group)
    axs[0].set(xscale='log',xlabel='Original resolution (megapixels, log scale)',ylabel='Image source',title='Resolution distributions');axs[0].invert_yaxis();axs[0].grid(axis='x',alpha=.2)
    axs[1].set(xscale='log',yscale='log',xlabel='Original width (pixels, log scale)',ylabel='Original height (pixels, log scale)',title='Dimensions and image shape');axs[1].grid(alpha=.2)
    axs[1].legend(fontsize=8,loc='upper left')
    fig.suptitle('Original image dimensions differ by source',fontsize=16,fontweight='bold')
    fig.savefig(CLEANED / 'plot2.png',dpi=180);plt.close(fig)

def summary(df):
    records=[]
    for group in GROUPS:
        s=df[df.group.eq(group)]
        records.append(dict(group=group,n=len(s),width_min=int(s.width.min()),width_median=float(s.width.median()),width_max=int(s.width.max()),height_min=int(s.height.min()),height_median=float(s.height.median()),height_max=int(s.height.max()),megapixels_median=float(s.megapixels.median()),square_pct=float(s.width.eq(s.height).mean()*100),formats=s['format'].value_counts().to_dict(),modes=s.color_mode.value_counts().to_dict()))
    (CLEANED / 'eda_summary.json').write_text(json.dumps(records,indent=2))
    df.groupby(['assigned_split','group']).size().unstack(fill_value=0).to_csv(CLEANED / 'split_counts.csv')
    return pd.DataFrame(records)

if __name__=='__main__':
    df=load_manifest();print(json.dumps(validate(df),indent=2));plots(df);print(summary(df).to_string(index=False))
    print('\nImages by split and generator (real photographs have no generator):')
    print(pd.crosstab(df.assigned_split,df.group).reindex(index=SPLIT_COUNTS,columns=GROUPS).to_string())
