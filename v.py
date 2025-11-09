#!/usr/bin/env python
"""
Verify that video paths in config_scaled.json are correct
Run this before preprocessing to catch path issues early
"""

import json
import glob
from pathlib import Path

def verify_config_paths(config_file='config_scaled.json'):
    """Verify all video paths exist and have videos"""
    
    print("=" * 80)
    print("VERIFYING VIDEO PATHS IN CONFIG")
    print("=" * 80 + "\n")
    
    # Load config
    try:
        with open(config_file) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"✗ ERROR: {config_file} not found!")
        return False
    
    all_good = True
    total_videos = 0
    
    # Check each category
    for category, path_patterns in config['categorized_video_paths'].items():
        print(f"{category.upper()}:")
        
        if not path_patterns or path_patterns == []:
            print(f"  ✗ ERROR: No paths specified for {category}")
            all_good = False
            continue
        
        category_videos = []
        for pattern in path_patterns:
            videos = glob.glob(pattern)
            category_videos.extend(videos)
            
            print(f"  Pattern: {pattern}")
            print(f"  Found: {len(videos)} videos")
            
            if len(videos) == 0:
                print(f"  ✗ WARNING: No videos found for this pattern!")
                all_good = False
            else:
                print(f"  ✓ Sample: {Path(videos[0]).name}")
        
        total_videos += len(category_videos)
        expected = config['dataset_stats'].get(category, 0)
        
        if len(category_videos) < expected:
            print(f"  ⚠️  WARNING: Expected {expected}, found {len(category_videos)}")
        elif len(category_videos) == expected:
            print(f"  ✓ Correct count: {len(category_videos)}/{expected}")
        else:
            print(f"  ⚠️  INFO: Found {len(category_videos)} (expected {expected})")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total videos found: {total_videos}")
    print(f"Expected total: {config['total_videos']}")
    
    if all_good and total_videos >= config['total_videos']:
        print("\n✓ All paths verified! Ready for preprocessing.")
        return True
    elif total_videos > 0:
        print("\n⚠️  Some paths have issues, but videos were found.")
        print("   You can proceed, but some categories might be incomplete.")
        return True
    else:
        print("\n✗ ERROR: No videos found! Check your paths.")
        print("\nCommon issues:")
        print("  1. Check dataset_path is correct")
        print("  2. Verify Kaggle dataset is downloaded")
        print("  3. Ensure folder structure matches config paths")
        return False


if __name__ == "__main__":
    success = verify_config_paths('config_scaled.json')
    
    if success:
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Update CATEGORY variable in 02_preprocess_frames_original.py")
        print("2. Run: python 02_preprocess_frames_original.py")
        print("3. Check visualization: preprocessing_sample_original.png")
    else:
        print("\n⚠️  Fix config paths before running preprocessing!")