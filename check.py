# import pandas as pd

# files = [
#     'preprocessed_frames_original.parquet',
#     'preprocessed_frames_deepfakes.parquet',
#     'preprocessed_frames_face2face.parquet',
#     'preprocessed_frames_faceswap.parquet',
#     'preprocessed_frames_neuraltextures.parquet',
#     'attacked_frames_clean.parquet'
# ]

# for fname in files:
#     try:
#         df = pd.read_parquet(fname)
#         n_videos = df['video_id'].nunique()
#         n_frames = len(df)
#         print(f"{fname}:")
#         print(f"  Unique videos: {n_videos}")
#         print(f"  Total frames: {n_frames}")
#         print("="*40)
#     except Exception as e:
#         print(f"{fname}: Could not read ({e})")

import pandas as pd

# Check video IDs in each category
for cat in ['original', 'deepfakes', 'face2face', 'faceswap', 'neuraltextures']:
    file = f'preprocessed_frames_{cat}.parquet'
    try:
        df = pd.read_parquet(file)
        vids = df['video_id'].unique()[:5]  # First 5
        print(f"{cat}:")
        print(f"  Sample IDs: {vids}\n")
    except:
        print(f"{cat}: Not found\n")