"""
Visualize Frames - FIXED for List Storage Issue

Handles both list-type and object-type storage in parquet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_image_from_row(row):
    """Safely extract image from parquet row"""
    img_data = row['image']
    
    # Handle different storage formats
    if isinstance(img_data, (list, np.ndarray)):
        img_flat = np.array(img_data)
    elif hasattr(img_data, '__iter__') and not isinstance(img_data, str):
        img_flat = np.array(list(img_data))
    else:
        # Single value - something went wrong
        print(f"⚠ Warning: Image data has wrong type: {type(img_data)}, size: {np.array(img_data).size}")
        return None
    
    # Check size
    expected_size = 224 * 224 * 3
    if img_flat.size != expected_size:
        print(f"⚠ Warning: Image size {img_flat.size}, expected {expected_size}")
        return None
    
    return img_flat.reshape((224, 224, 3)).astype(np.uint8)

def visualize_frames_from_parquet(parquet_file, video_id=None, num_frames=10, output_file='frame_visualization.png'):
    """Visualize frames from a specific video"""
    
    if not Path(parquet_file).exists():
        print(f"✗ {parquet_file} not found")
        return
    
    print(f"Loading {parquet_file}...")
    
    # Try different engines
    try:
        df = pd.read_parquet(parquet_file, engine='pyarrow')
    except:
        try:
            df = pd.read_parquet(parquet_file, engine='fastparquet')
        except:
            print("✗ Failed to load with both engines")
            return
    
    print(f"✓ Loaded {len(df)} frames\n")
    
    # Debug: Check image column type
    sample_img = df.iloc[0]['image']
    print(f"Debug: Image column type: {type(sample_img)}")
    print(f"Debug: Image data size: {np.array(sample_img).size}")
    
    unique_videos = df['video_id'].unique()
    print(f"Available videos: {len(unique_videos)}")
    
    if video_id is None:
        video_id = unique_videos[0]
        print(f"Using first video: {video_id}")
    else:
        if video_id not in df['video_id'].values:
            print(f"✗ Video {video_id} not found")
            return
    
    video_frames = df[df['video_id'] == video_id].head(num_frames)
    print(f"✓ Found {len(video_frames)} frames for video {video_id}\n")
    
    label = video_frames.iloc[0]['label']
    manip_type = video_frames.iloc[0]['manipulation_type']
    label_text = "Real" if label == 0 else "Fake"
    
    attack_type = "Clean"
    if "fgsm" in parquet_file.lower():
        attack_type = "FGSM"
    elif "pgd" in parquet_file.lower():
        attack_type = "PGD"
    elif "malafide" in parquet_file.lower():
        attack_type = "2D-Malafide"
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Video: {video_id} | Attack: {attack_type}\nLabel: {label_text} | Type: {manip_type}', 
                 fontsize=14, fontweight='bold')
    
    valid_count = 0
    for idx, ax in enumerate(axes.flat):
        if idx < len(video_frames):
            img = load_image_from_row(video_frames.iloc[idx])
            
            if img is not None:
                ax.imshow(img)
                ax.set_title(f'Frame {idx+1}', fontsize=10)
                ax.axis('off')
                valid_count += 1
            else:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.axis('off')
        else:
            ax.axis('off')
    
    if valid_count == 0:
        print("✗ No valid frames could be loaded")
        plt.close()
        return
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {valid_count} frames to {output_file}")
    plt.close()

def main():
    print("=" * 80)
    print("FRAME VISUALIZATION - DEBUG MODE")
    print("=" * 80 + "\n")
    
    # Test with clean file first
    if Path('attacked_frames_pgd.parquet').exists():
        print("[TEST] Checking clean parquet structure...\n")
        visualize_frames_from_parquet(
            'attacked_frames_pgd.parquet',
            video_id=None,
            num_frames=10,
            output_file='test_clean.png'
        )
    else:
        print("✗ attacked_frames_clean.parquet not found")
        print("\nTrying original preprocessed files...")
        
        if Path('preprocessed_frames_original.parquet').exists():
            print("\n[TEST] Using original preprocessed file...\n")
            visualize_frames_from_parquet(
                'preprocessed_frames_original.parquet',
                video_id=None,
                num_frames=10,
                output_file='test_original.png'
            )

if __name__ == "__main__":
    main()