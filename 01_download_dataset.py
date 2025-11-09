"""
STEP 1: Download FaceForensics++ C23 - CORRECTED VERSION

Key Fix: Maintains correspondence between original video and its 5 variants
- Groups videos by source name
- Ensures each video has all 5 manipulation types
- Validates dataset structure
"""

import json
import kagglehub
from pathlib import Path
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def download_dataset():
    """Download FaceForensics++ C23 from Kaggle"""
    print("[INFO] Downloading FaceForensics++ C23 from Kaggle...")
    dataset_path = kagglehub.dataset_download("xdxd003/ff-c23")
    print(f"✓ Downloaded to: {dataset_path}\n")
    return dataset_path

def find_base_directory(dataset_path):
    """Find the base directory containing video folders"""
    dataset_path = Path(dataset_path)
    
    # Try multiple possible structures
    possible_bases = [
        dataset_path / "FaceForensics++_C23",
        dataset_path / "FaceForensics_C23",
        dataset_path / "FaceForensics",
        dataset_path
    ]
    
    for base in possible_bases:
        if base.exists():
            # Check if it has the expected folders
            if (base / "original").exists() or (base / "Deepfakes").exists():
                print(f"✓ Found base directory: {base}\n")
                return base
    
    print(f"✗ Could not find FaceForensics base directory")
    print(f"  Tried: {possible_bases}")
    return None

def load_videos_by_category(base_path):
    """Load videos from each category folder"""
    
    categories = {
        'original': 'original',
        'deepfakes': 'Deepfakes',
        'face2face': 'Face2Face',
        'faceswap': 'FaceSwap',
        'neuraltextures': 'NeuralTextures'
    }
    
    print("=" * 80)
    print("SCANNING DIRECTORY STRUCTURE")
    print("=" * 80 + "\n")
    
    categorized = {}
    
    for key, folder_name in categories.items():
        folder_path = base_path / folder_name
        
        if folder_path.exists():
            videos = sorted(list(folder_path.glob("*.mp4")))
            categorized[key] = videos
            label = "REAL" if key == 'original' else "FAKE"
            print(f"{key.upper():20s}: {len(videos):4d} videos ({label})")
        else:
            print(f"{key.upper():20s}: ✗ FOLDER NOT FOUND")
            categorized[key] = []
    
    total = sum(len(v) for v in categorized.values())
    print(f"\n{'TOTAL':20s}: {total:4d} videos\n")
    
    return categorized

def group_videos_by_source(categorized_videos):
    """
    Group videos by their source name to establish correspondence.
    
    Expected: Each video should appear in ALL 5 categories with the same base name.
    Example:
      original/video_001.mp4
      deepfakes/video_001.mp4
      face2face/video_001.mp4
      faceswap/video_001.mp4
      neuraltextures/video_001.mp4
    """
    
    print("=" * 80)
    print("GROUPING VIDEOS BY SOURCE")
    print("=" * 80 + "\n")
    
    # Extract base names
    video_names = {}
    
    for category, videos in categorized_videos.items():
        for video_path in videos:
            video_name = video_path.stem  # Remove .mp4 extension
            
            if video_name not in video_names:
                video_names[video_name] = {}
            
            video_names[video_name][category] = str(video_path)
    
    print(f"Found {len(video_names)} unique video sources\n")
    
    # Validate completeness
    print("=" * 80)
    print("VALIDATING DATA COMPLETENESS")
    print("=" * 80 + "\n")
    
    complete_videos = {}
    incomplete_videos = {}
    
    for video_name, variants in video_names.items():
        has_all_5 = len(variants) == 5
        
        if has_all_5:
            complete_videos[video_name] = variants
        else:
            incomplete_videos[video_name] = variants
    
    print(f"✓ Complete videos (all 5 variants): {len(complete_videos)}")
    print(f"✗ Incomplete videos (missing variants): {len(incomplete_videos)}\n")
    
    # Show incomplete videos sample
    if incomplete_videos:
        print("[WARNING] Sample of incomplete videos:")
        for i, (name, variants) in enumerate(list(incomplete_videos.items())[:5]):
            print(f"  {name}: {list(variants.keys())}")
        print()
    
    return complete_videos, incomplete_videos, video_names

def build_correspondence_config(complete_videos, categorized_videos):
    """Build config with proper video-to-variant correspondence"""
    
    print("=" * 80)
    print("BUILDING CORRESPONDENCE CONFIG")
    print("=" * 80 + "\n")
    
    # Separate by label
    config_data = {
        'original': [],      # label=0 (Real)
        'deepfakes': [],     # label=1 (Fake)
        'face2face': [],     # label=1 (Fake)
        'faceswap': [],      # label=1 (Fake)
        'neuraltextures': [] # label=1 (Fake)
    }
    
    for video_name, variants in complete_videos.items():
        for category in config_data.keys():
            if category in variants:
                config_data[category].append(variants[category])
    
    # Statistics
    stats = {k: len(v) for k, v in config_data.items()}
    
    print(f"Dataset Statistics:")
    print(f"  Original (Real):  {stats['original']:4d} videos")
    print(f"  Deepfakes (Fake): {stats['deepfakes']:4d} videos")
    print(f"  Face2Face (Fake): {stats['face2face']:4d} videos")
    print(f"  FaceSwap (Fake):  {stats['faceswap']:4d} videos")
    print(f"  NeuralTextures:   {stats['neuraltextures']:4d} videos")
    print(f"\n  Real:  {stats['original']}")
    print(f"  Fake:  {stats['deepfakes'] + stats['face2face'] + stats['faceswap'] + stats['neuraltextures']}")
    print(f"  TOTAL: {sum(stats.values())}")
    print(f"\n  Expected frames (10/video): {sum(stats.values()) * 10}\n")
    
    return config_data, stats

def save_config(config_data, stats, dataset_path):
    """Save configuration file"""
    
    config = {
        'dataset_path': str(dataset_path),
        'dataset_stats': stats,
        'total_videos': sum(stats.values()),
        'total_frames_expected': sum(stats.values()) * 10,
        'random_seed': RANDOM_SEED,
        'categorized_video_paths': config_data,
        'important_note': 'Videos are grouped by source - each video has all 5 variants'
    }
    
    with open('config_scaled.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("=" * 80)
    print("CONFIG SAVED")
    print("=" * 80)
    print(f"✓ Saved: config_scaled.json\n")

def main():
    print("\n" + "=" * 80)
    print("STEP 1: DOWNLOAD FACEFORENSICS++ C23 - CORRECTED")
    print("=" * 80 + "\n")
    
    # Download
    dataset_path = download_dataset()
    
    # Find base directory
    base_path = find_base_directory(dataset_path)
    if base_path is None:
        print("✗ Failed to find dataset structure")
        return
    
    # Load videos by category
    categorized_videos = load_videos_by_category(base_path)
    
    # Group by source
    complete_videos, incomplete_videos, all_videos = group_videos_by_source(categorized_videos)
    
    if not complete_videos:
        print("\n⚠ WARNING: No complete video groups found!")
        print("The dataset may have a different structure.")
        print("Proceeding with individual files...\n")
        config_data, stats = build_correspondence_config(all_videos, categorized_videos)
    else:
        config_data, stats = build_correspondence_config(complete_videos, categorized_videos)
    
    # Save config
    save_config(config_data, stats, dataset_path)
    
    print("=" * 80)
    print("✓ STEP 1 COMPLETE")
    print("=" * 80)
    print(f"\n✓ Next: python 02_preprocess_frames_fast_original.py (and others)\n")

if __name__ == "__main__":
    main()