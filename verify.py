import pandas as pd

files = [
    'preprocessed_frames_original.parquet',
    'preprocessed_frames_deepfakes.parquet',
    'preprocessed_frames_face2face.parquet',
    'preprocessed_frames_faceswap.parquet',
    'preprocessed_frames_neuraltextures.parquet'
]

print("=" * 80)
print("CHECKING IF EACH MANIPULATION TYPE IS PRESERVED")
print("=" * 80 + "\n")

for f in files:
    df = pd.read_parquet(f)
    vid_sample = df['video_id'].iloc[0]
    manip_sample = df['manipulation_type'].iloc[0]
    label_sample = df['label'].iloc[0]
    
    print(f"{f}:")
    print(f"  Sample video_id: {vid_sample}")
    print(f"  Sample manipulation_type: {manip_sample}")
    print(f"  Sample label: {label_sample}")
    print(f"  All manipulation_types: {df['manipulation_type'].unique()}")
    print()

# Combined
print("=" * 80)
print("AFTER COMBINING")
print("=" * 80 + "\n")

dfs = [pd.read_parquet(f) for f in files]
df_combined = pd.concat(dfs, ignore_index=True)

print(f"Total frames: {len(df_combined)}")
print(f"\nManipulation type breakdown:")
print(df_combined['manipulation_type'].value_counts())

print(f"\nLabel breakdown:")
print(df_combined['label'].value_counts())

# Check if same video_id exists across multiple manipulation types
print(f"\n" + "=" * 80)
print("CHECKING VIDEO_ID ACROSS MANIPULATION TYPES")
print("=" * 80 + "\n")

# Pick a video_id
vid = df_combined['video_id'].iloc[0]

rows_with_vid = df_combined[df_combined['video_id'] == vid]
print(f"Video ID: {vid}")
print(f"Found in {len(rows_with_vid)} frames")
print(f"Manipulation types for this video:")
print(rows_with_vid[['video_id', 'manipulation_type', 'label']].to_string())
