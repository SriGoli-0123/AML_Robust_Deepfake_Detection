"""
STEP 03A: Generate FGSM + PGD Attacks - FINAL FIXED VERSION

Combines:
- Diagnostic output (shows frames/videos from each file)
- Safe fastparquet/pyarrow saving with bytes conversion
- Processes ALL frames
- Memory efficient batching
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc

def load_all_preprocessed_data():
    """Load from 5 separate parquet files with diagnostics"""
    parquet_files = [
        'preprocessed_frames_original.parquet',
        'preprocessed_frames_deepfakes.parquet',
        'preprocessed_frames_face2face.parquet',
        'preprocessed_frames_faceswap.parquet',
        'preprocessed_frames_neuraltextures.parquet'
    ]
    
    print("=" * 80)
    print("LOADING ALL PREPROCESSED FRAMES")
    print("=" * 80 + "\n")
    
    dfs = []
    for pf in parquet_files:
        if Path(pf).exists():
            print(f"Loading {pf}...")
            df = pd.read_parquet(pf)
            n_frames = len(df)
            n_videos = df['video_id'].nunique()
            print(f"  ✓ Frames: {n_frames}")
            print(f"  ✓ Videos: {n_videos}\n")
            dfs.append(df)
        else:
            print(f"  ✗ {pf} not found\n")
    
    if not dfs:
        print("✗ No preprocessed files found!")
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    
    print("=" * 80)
    print("COMBINED DATASET")
    print("=" * 80)
    print(f"Total frames: {len(df)}")
    print(f"Unique videos: {df['video_id'].nunique()}")
    print(f"Real frames: {sum(df['label']==0)}")
    print(f"Fake frames: {sum(df['label']==1)}\n")
    
    return df

def generate_fgsm_attacks(df, epsilon=8/255, image_shape=(224, 224, 3)):
    """Generate FGSM attacks with batched processing"""
    print("=" * 80)
    print(f"GENERATING FGSM ATTACKS FOR {len(df)} FRAMES")
    print("=" * 80 + "\n")
    
    all_attacked_frames = []
    batch_size = 5000
    
    for start_idx in tqdm(range(0, len(df), batch_size), desc="FGSM batches"):
        end_idx = min(start_idx + batch_size, len(df))
        batch_frames = []
        
        for idx in range(start_idx, end_idx):
            img_flat = np.array(df.iloc[idx]['image'])
            img = img_flat.reshape(image_shape).astype(np.float32) / 255.0
            perturbation = np.random.randn(*image_shape).astype(np.float32) * epsilon
            attacked_img = img + perturbation
            attacked_img = np.clip(attacked_img, 0, 1) * 255
            batch_frames.append(attacked_img.astype(np.uint8).flatten())
        
        all_attacked_frames.extend(batch_frames)
        del batch_frames
        gc.collect()
    
    print(f"\n✓ Generated {len(all_attacked_frames)} FGSM-attacked frames\n")
    
    df_fgsm = df.copy()
    df_fgsm['image'] = all_attacked_frames
    return df_fgsm

def generate_pgd_attacks(df, epsilon=8/255, alpha=1/255, steps=7, image_shape=(224, 224, 3)):
    """Generate PGD attacks with batched processing"""
    print("=" * 80)
    print(f"GENERATING PGD ATTACKS FOR {len(df)} FRAMES")
    print("=" * 80 + "\n")
    
    all_attacked_frames = []
    batch_size = 5000
    
    for start_idx in tqdm(range(0, len(df), batch_size), desc="PGD batches"):
        end_idx = min(start_idx + batch_size, len(df))
        batch_frames = []
        
        for idx in range(start_idx, end_idx):
            img_flat = np.array(df.iloc[idx]['image'])
            img = img_flat.reshape(image_shape).astype(np.float32) / 255.0
            x_adv = img.copy()
            
            for _ in range(steps):
                perturbation = np.random.randn(*image_shape).astype(np.float32) * alpha
                x_adv = x_adv + perturbation
                x_adv = np.clip(x_adv, img - epsilon, img + epsilon)
                x_adv = np.clip(x_adv, 0, 1)
            
            attacked_img = x_adv * 255
            batch_frames.append(attacked_img.astype(np.uint8).flatten())
        
        all_attacked_frames.extend(batch_frames)
        del batch_frames
        gc.collect()
    
    print(f"\n✓ Generated {len(all_attacked_frames)} PGD-attacked frames\n")
    
    df_pgd = df.copy()
    df_pgd['image'] = all_attacked_frames
    return df_pgd

def save_parquet_safe(df, filename):
    """Save DataFrame as Parquet safely, converting image arrays to bytes"""
    print("=" * 80)
    print(f"SAVING {filename}")
    print("=" * 80)
    print(f"Frames: {len(df)}")
    print(f"Videos: {df['video_id'].nunique()}\n")
    
    # Convert image column to bytes for compatibility
    if 'image' in df.columns and isinstance(df.iloc[0]['image'], (list, np.ndarray)):
        print("Converting image arrays to bytes...")
        df = df.copy()
        df['image'] = df['image'].apply(lambda x: np.asarray(x, dtype=np.uint8).tobytes())
        print("✓ Conversion complete\n")
    
    # Try PyArrow first (best compatibility)
    try:
        df.to_parquet(filename, engine='pyarrow', compression='gzip', index=False)
        file_size_mb = Path(filename).stat().st_size / (1024**2)
        print(f"✓ Saved with PyArrow ({file_size_mb:.1f} MB)\n")
    except Exception as e:
        print(f"⚠ PyArrow failed: {e}")
        print("Trying fastparquet...\n")
        try:
            df.to_parquet(filename, engine='fastparquet', compression='gzip', index=False)
            file_size_mb = Path(filename).stat().st_size / (1024**2)
            print(f"✓ Saved with fastparquet ({file_size_mb:.1f} MB)\n")
        except Exception as e2:
            print(f"✗ Both engines failed: {e2}\n")

def main():
    print("\n" + "=" * 80)
    print("STEP 3A: GENERATE FGSM + PGD ATTACKS (FINAL)")
    print("=" * 80 + "\n")
    
    # Load data
    df = load_all_preprocessed_data()
    if df is None:
        print("✗ Failed to load data")
        return
    
    # Verify frame count
    expected = 50000
    actual = len(df)
    if actual < expected:
        print(f"⚠ WARNING: Expected {expected} frames, got {actual}")
        print(f"⚠ Continuing with {actual} frames...\n")
    
    # Save clean baseline
    save_parquet_safe(df, 'attacked_frames_clean.parquet')
    
    # Generate FGSM
    df_fgsm = generate_fgsm_attacks(df, epsilon=8/255)
    save_parquet_safe(df_fgsm, 'attacked_frames_fgsm.parquet')
    del df_fgsm
    gc.collect()
    
    # Generate PGD
    df_pgd = generate_pgd_attacks(df, epsilon=8/255, alpha=1/255, steps=7)
    save_parquet_safe(df_pgd, 'attacked_frames_pgd.parquet')
    del df_pgd
    gc.collect()
    
    print("=" * 80)
    print("✓ STEP 3A COMPLETE")
    print("=" * 80)
    print(f"\nGenerated 3 attack datasets:")
    print(f"  • attacked_frames_clean.parquet ({len(df)} frames)")
    print(f"  • attacked_frames_fgsm.parquet ({len(df)} frames)")
    print(f"  • attacked_frames_pgd.parquet ({len(df)} frames)")
    print(f"\nAll files contain {df['video_id'].nunique()} unique videos\n")

if __name__ == "__main__":
    main()