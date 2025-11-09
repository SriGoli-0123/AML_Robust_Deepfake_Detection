"""
View Video Frames: Original + Its Deepfake Variants

Shows 1 original video + all manipulated variants that use it as source
Layout: 10 rows (frames) × up to 5 columns (1 original + deepfakes using it as source)

Example:
  Original 000
  Deepfakes: 000_003, 000_123, 000_456, etc. (all derived from 000)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_parquets():
    """Load all 5 preprocessed parquet files"""
    files = {
        'original': 'preprocessed_frames_original.parquet',
        'deepfakes': 'preprocessed_frames_deepfakes.parquet',
        'face2face': 'preprocessed_frames_face2face.parquet',
        'faceswap': 'preprocessed_frames_faceswap.parquet',
        'neuraltextures': 'preprocessed_frames_neuraltextures.parquet'
    }
    
    dfs = {}
    print("Loading parquet files...\n")
    
    for category, file in files.items():
        if Path(file).exists():
            df = pd.read_parquet(file)
            dfs[category] = df
            print(f"✓ {category:15s}: {len(df)} frames")
        else:
            print(f"✗ {category:15s}: Not found")
    
    return dfs

def get_source_id(video_id):
    """Extract source ID from video_id
    
    Examples:
      '000_original' → '000'
      '000_003_deepfakes' → '000' (first number)
      '001_870_face2face' → '001'
    """
    parts = video_id.split('_')
    return parts[0]

def find_variants_for_original(dfs, original_source_id):
    """
    Find all manipulated variants that use this original as source
    
    For original '000', find:
      - deepfakes: 000_xxx_deepfakes
      - face2face: 000_xxx_face2face
      - faceswap: 000_xxx_faceswap
      - neuraltextures: 000_xxx_neuraltextures
    """
    
    variants = {}
    
    # Get original frames
    if 'original' in dfs:
        original_df = dfs['original']
        original_frames = original_df[original_df['video_id'] == f"{original_source_id}_original"]
        
        if len(original_frames) > 0:
            variants['original'] = original_frames
            print(f"  original:       {len(original_frames)} frames")
        else:
            print(f"  original:       Not found")
    
    # Get manipulated variants that use this source
    for category in ['deepfakes', 'face2face', 'faceswap', 'neuraltextures']:
        if category in dfs:
            df = dfs[category]
            
            # Find all videos that start with this source ID
            matching_videos = df[df['video_id'].str.startswith(f"{original_source_id}_")]
            
            if len(matching_videos) > 0:
                variants[category] = matching_videos
                unique_targets = matching_videos['video_id'].nunique()
                print(f"  {category:15s}: {len(matching_videos)} frames from {unique_targets} manipulation(s)")
            else:
                print(f"  {category:15s}: Not found")
    
    return variants

def visualize_original_with_variants(variants, source_id, output_file='original_with_variants.png'):
    """
    Visualize original + its manipulated variants
    
    Layout:
      Row 1 (Frame 1): [Original 000] [Deepfake 000_003] [Face2Face 000_123] [etc...]
      Row 2 (Frame 2): [Original 000] [Deepfake 000_003] [Face2Face 000_123] [etc...]
      ...
      Row 10 (Frame 10): [Original 000] [Deepfake 000_003] [Face2Face 000_123] [etc...]
    """
    
    categories = ['original', 'deepfakes', 'face2face', 'faceswap', 'neuraltextures']
    available_categories = [cat for cat in categories if cat in variants]
    num_frames = 10
    
    if not available_categories:
        print("✗ No data found")
        return
    
    # Create figure
    fig, axes = plt.subplots(num_frames, len(available_categories), 
                             figsize=(len(available_categories)*3, num_frames*3))
    
    # Handle single column case
    if len(available_categories) == 1:
        axes = axes.reshape(-1, 1)
    
    title = f'Original Video: {source_id}\nWith All Manipulation Variants Using It as Source'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Column headers
    for col_idx, category in enumerate(available_categories):
        ax = axes[0, col_idx]
        label = "REAL" if category == 'original' else "FAKE"
        
        # Show variant count for manipulated
        if category != 'original':
            variant_count = variants[category]['video_id'].nunique()
            title_str = f'{category.upper()}\n({label})\n{variant_count} variant(s)'
        else:
            title_str = f'{category.upper()}\n({label})'
        
        ax.set_title(title_str, fontsize=11, fontweight='bold')
    
    # Plot frames row by row
    for frame_idx in range(num_frames):
        for col_idx, category in enumerate(available_categories):
            ax = axes[frame_idx, col_idx]
            
            # Get frame data
            if frame_idx < len(variants[category]):
                try:
                    img_flat = np.array(variants[category].iloc[frame_idx]['image'])
                    
                    # Handle bytes vs array
                    if isinstance(img_flat, bytes):
                        img_flat = np.frombuffer(img_flat, dtype=np.uint8)
                    
                    img = img_flat.reshape((224, 224, 3)).astype(np.uint8)
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {str(e)[:10]}', ha='center', va='center', fontsize=8)
                
                # Row label on left
                if col_idx == 0:
                    ax.set_ylabel(f'Frame {frame_idx+1}', fontsize=11, fontweight='bold', 
                                 rotation=0, labelpad=40, va='center')
                
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.show()

def list_original_videos(dfs):
    """List all original video source IDs"""
    
    if 'original' not in dfs:
        print("\n✗ No original parquet found")
        return []
    
    df = dfs['original']
    source_ids = []
    
    for vid in df['video_id'].unique():
        source_id = get_source_id(vid)
        source_ids.append(source_id)
    
    source_ids = sorted(list(set(source_ids)))
    
    print("\n" + "="*80)
    print("AVAILABLE ORIGINAL VIDEO SOURCE IDS")
    print("="*80)
    print(f"\nFound {len(source_ids)} original videos\n")
    print("First 20:")
    for i, sid in enumerate(source_ids[:20]):
        print(f"  {i+1:3d}. {sid}")
    
    return source_ids

def main():
    print("="*80)
    print("ORIGINAL VIDEO + ITS MANIPULATION VARIANTS VIEWER")
    print("="*80 + "\n")
    
    # Load all parquets
    dfs = load_all_parquets()
    
    if not dfs:
        print("\n✗ No parquet files found")
        return
    
    # List available originals
    source_ids = list_original_videos(dfs)
    
    if not source_ids:
        return
    
    # Use first original (or specify any)
    original_source_id = source_ids[0]  # Change this to view different original
    
    print(f"\n" + "="*80)
    print(f"VISUALIZING ORIGINAL: {original_source_id}")
    print("="*80)
    print(f"\nSearching for variants using source {original_source_id}...\n")
    
    # Find all variants
    variants = find_variants_for_original(dfs, original_source_id)
    
    # Visualize
    if variants:
        visualize_original_with_variants(
            variants, 
            original_source_id, 
            output_file=f'original_{original_source_id}_with_variants.png'
        )
    else:
        print(f"\n✗ No variants found for {original_source_id}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80 + "\n")
    print("Tips:")
    print(f"  - Change line ~180: original_source_id = '{source_ids[5]}' to view different original")
    print(f"  - Available originals: {source_ids[:5]} ... {source_ids[-5:]}\n")

if __name__ == "__main__":
    main()