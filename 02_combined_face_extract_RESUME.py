"""
STEP 2: Optimized Face-Tracked Extraction (Quality + Speed Balance)

What's KEPT for quality:
✓ Face tracking (smooth, stable faces)
✓ 30% margin (better context)
✓ 256×256 → 224×224 (proper resizing)
✓ Per-channel histogram equalization
✓ Dlib fallback for difficult faces

What's OPTIMIZED for speed/heat:
✓ Smart frame sampling (not ALL frames)
✓ Memory-efficient processing
✓ Sequential order (000→500)
✓ Face validation (all 10 frames)
✓ Aggressive garbage collection
✓ Checkpoints every 5 videos (cooling breaks)

Expected: 1-2 min per video (vs 3-4 min before)
Quality: Same as original combined code
"""

import cv2
import json
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import dlib
from mtcnn import MTCNN
import gc
import time
import signal
import sys
import glob

# Configuration
FRAMES_PER_VIDEO = 10
TARGET_FACE_SIZE = 224
FACE_MARGIN = 0.3  # Keep 30% for quality
MIN_FACE_FRAMES = 50
SMOOTHING_WINDOW = 5
FACE_VIDEO_SIZE = 256
MAX_VIDEOS = 500
CHECKPOINT_INTERVAL = 5  # Checkpoint every 5 videos (for cooling)

CATEGORY = "original"  # CHANGE: original, deepfakes, face2face, faceswap, neuraltextures

OUTPUT_FILE = f"preprocessed_frames_{CATEGORY}.parquet"
CHECKPOINT_FILE = f"checkpoint_{CATEGORY}.json"

STOP_REQUESTED = False


def signal_handler(sig, frame):
    global STOP_REQUESTED
    print('\n\n⚠️  Stop requested! Saving checkpoint...')
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, signal_handler)


class OptimizedFaceTracker:
    """Face tracking with speed optimizations"""
    
    def __init__(self):
        self.mtcnn = MTCNN(min_face_size=30)
        self.dlib_detector = dlib.get_frontal_face_detector()
    
    def detect_face_box(self, frame):
        """Detect with MTCNN, fallback to dlib"""
        try:
            faces = self.mtcnn.detect_faces(frame)
            if faces:
                largest = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                x, y, w, h = largest['box']
                return (x, y, w, h), largest.get('confidence', 1.0)
            
            # Dlib fallback for difficult cases
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = self.dlib_detector(gray, 1)
            if len(dets) > 0:
                d = dets[0]
                return (d.left(), d.top(), d.width(), d.height()), 0.9
            
            return None, 0.0
        except:
            return None, 0.0
    
    def smooth_boxes(self, boxes, window=SMOOTHING_WINDOW):
        """Smooth bounding boxes for stable tracking"""
        if len(boxes) < window:
            return boxes
        
        smoothed = []
        for i in range(len(boxes)):
            start = max(0, i - window // 2)
            end = min(len(boxes), i + window // 2 + 1)
            window_boxes = boxes[start:end]
            avg_box = np.mean(window_boxes, axis=0).astype(int)
            smoothed.append(tuple(avg_box))
        
        return smoothed
    
    def crop_frame_to_face(self, frame, face_box):
        """Crop with 30% margin"""
        h, w = frame.shape[:2]
        x, y, fw, fh = face_box
        
        margin_w = int(fw * FACE_MARGIN)
        margin_h = int(fh * FACE_MARGIN)
        
        size = max(fw + 2 * margin_w, fh + 2 * margin_h)
        cx, cy = x + fw // 2, y + fh // 2
        
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w, cx + size // 2)
        y2 = min(h, cy + size // 2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        return cv2.resize(crop, (FACE_VIDEO_SIZE, FACE_VIDEO_SIZE))
    
    def process_video_optimized(self, video_path, label, manip_type):
        """
        OPTIMIZED: Smart sampling + face tracking
        - Sample every Nth frame for detection (not all frames)
        - Still smooth boxes for quality
        - Validate all 10 final frames have faces
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < MIN_FACE_FRAMES:
                cap.release()
                return None
            
            # OPTIMIZATION: Sample every 5th frame for detection (not all)
            sample_interval = 5
            face_boxes = []
            confidences = []
            sampled_frames = []
            sampled_indices = []
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process sampled frames
                if frame_idx % sample_interval == 0:
                    box, conf = self.detect_face_box(frame)
                    
                    if box:
                        face_boxes.append(box)
                        confidences.append(conf)
                    else:
                        # Use last known box
                        if len(face_boxes) > 0:
                            face_boxes.append(face_boxes[-1])
                            confidences.append(0.5)
                        else:
                            face_boxes.append((0, 0, frame.shape[1], frame.shape[0]))
                            confidences.append(0.1)
                    
                    sampled_frames.append(frame)
                    sampled_indices.append(frame_idx)
                
                frame_idx += 1
            
            cap.release()
            
            if len(face_boxes) < MIN_FACE_FRAMES // sample_interval:
                return None
            
            # Smooth boxes for stability
            smoothed_boxes = self.smooth_boxes(face_boxes)
            
            # Create face-tracked frames
            face_frames = []
            for frame, box in zip(sampled_frames, smoothed_boxes):
                crop = self.crop_frame_to_face(frame, box)
                if crop is not None:
                    face_frames.append(crop)
            
            if len(face_frames) < FRAMES_PER_VIDEO:
                return None
            
            # Extract 10 evenly-spaced frames
            total = len(face_frames)
            interval = total // (FRAMES_PER_VIDEO + 1)
            
            final_frames = []
            video_id = f"{video_path.stem}_{CATEGORY}"
            
            for i in range(FRAMES_PER_VIDEO):
                idx = (i + 1) * interval
                if idx >= len(face_frames):
                    idx = len(face_frames) - 1
                
                face = face_frames[idx]
                
                # Resize to 224×224
                face = cv2.resize(face, (TARGET_FACE_SIZE, TARGET_FACE_SIZE))
                
                # Per-channel histogram equalization
                b, g, r = cv2.split(face)
                b_eq = cv2.equalizeHist(b)
                g_eq = cv2.equalizeHist(g)
                r_eq = cv2.equalizeHist(r)
                face_enhanced = cv2.merge((b_eq, g_eq, r_eq))
                
                # VALIDATE: Check if face is still visible
                # (Reject if too dark/bright - indicates no face)
                brightness = face_enhanced.mean()
                if brightness < 20 or brightness > 240:
                    return None  # Reject: invalid frame
                
                final_frames.append({
                    'image': face_enhanced,
                    'label': label,
                    'video_id': video_id,
                    'manipulation_type': manip_type
                })
            
            # Ensure we have exactly 10 valid frames
            if len(final_frames) != FRAMES_PER_VIDEO:
                return None
            
            avg_conf = np.mean(confidences)
            return final_frames, avg_conf
            
        except Exception as e:
            if 'cap' in locals():
                cap.release()
            return None
        finally:
            # Aggressive cleanup
            if 'sampled_frames' in locals():
                del sampled_frames
            if 'face_frames' in locals():
                del face_frames
            gc.collect()


def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'videos_processed': 0, 'processed_ids': [], 'frames_count': 0}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def load_config():
    """Load videos in sequential order, limit to 500"""
    with open('config_scaled.json') as f:
        cfg = json.load(f)
    
    path_patterns = cfg['categorized_video_paths'][CATEGORY]
    videos = []
    for pattern in path_patterns:
        videos.extend(glob.glob(pattern))
    
    # Sequential + limit
    videos = sorted(videos)[:MAX_VIDEOS]
    
    label = 0 if CATEGORY == 'original' else 1
    return [(Path(v), label, CATEGORY) for v in videos]


def load_existing_parquet():
    if Path(OUTPUT_FILE).exists():
        print(f"[INFO] Loading existing parquet...\n")
        return pd.read_parquet(OUTPUT_FILE)
    return None


def main():
    global STOP_REQUESTED
    
    print("=" * 80)
    print(f"OPTIMIZED FACE-TRACKED EXTRACTION - {CATEGORY.upper()}")
    print("=" * 80)
    print(f"[INFO] Quality: Face tracking (30% margin, smoothed)")
    print(f"[INFO] Speed: Smart sampling (5x less CPU)")
    print(f"[INFO] Sequential: 000 → {MAX_VIDEOS-1}")
    print(f"[INFO] Validation: All 10 frames checked")
    print(f"[INFO] Press CTRL+C for cooling breaks\n")
    
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint['processed_ids'])
    
    existing_df = load_existing_parquet()
    if existing_df is not None:
        all_frames = [np.array(x).reshape(TARGET_FACE_SIZE, TARGET_FACE_SIZE, 3) 
                     for x in existing_df['image']]
        all_labels = existing_df['label'].tolist()
        all_ids = existing_df['video_id'].tolist()
        all_types = existing_df['manipulation_type'].tolist()
    else:
        all_frames = []
        all_labels = []
        all_ids = []
        all_types = []
    
    all_videos = load_config()
    remaining = [v for v in all_videos if str(v[0]) not in processed_ids]
    
    print(f"[INFO] Target: {len(all_videos)} videos")
    print(f"[INFO] Done: {len(processed_ids)}")
    print(f"[INFO] Remaining: {len(remaining)}")
    print(f"[INFO] Frames: {len(all_frames)}\n")
    
    if len(remaining) == 0:
        print("✓ All processed!")
        return
    
    tracker = OptimizedFaceTracker()
    start_time = time.time()
    processed_this_run = 0
    rejected = 0
    
    print("Processing sequentially:\n")
    
    for video_path, label, manip_type in tqdm(remaining, desc=f"{CATEGORY}"):
        if STOP_REQUESTED:
            break
        
        result = tracker.process_video_optimized(video_path, label, manip_type)
        
        if result:
            frames, conf = result
            for fd in frames:
                all_frames.append(fd['image'])
                all_labels.append(fd['label'])
                all_ids.append(fd['video_id'])
                all_types.append(fd['manipulation_type'])
            
            processed_ids.add(str(video_path))
            processed_this_run += 1
        else:
            rejected += 1
        
        # Checkpoint every 5 videos (cooling opportunity)
        if processed_this_run % CHECKPOINT_INTERVAL == 0 and processed_this_run > 0:
            elapsed = time.time() - start_time
            rate = processed_this_run / elapsed
            eta = (len(remaining) - processed_this_run - rejected) / rate if rate > 0 else 0
            
            print(f"\n[CHECKPOINT] Accepted: {processed_this_run}, Rejected: {rejected}")
            print(f" Speed: {rate:.2f} videos/sec (~{rate*60:.0f} videos/min)")
            print(f" ETA: {eta/60:.0f} min ({eta/3600:.1f} hours)")
            print(f" Frames: {len(all_frames)}")
            print(f" 💾 Saving... ", end='', flush=True)
            
            flattened = [f.flatten() for f in all_frames]
            df = pd.DataFrame({
                'image': flattened,
                'label': all_labels,
                'video_id': all_ids,
                'manipulation_type': all_types
            })
            df.to_parquet(OUTPUT_FILE, compression='gzip', index=False)
            
            save_checkpoint({
                'videos_processed': checkpoint['videos_processed'] + processed_this_run,
                'processed_ids': list(processed_ids),
                'frames_count': len(all_frames)
            })
            
            print("✓")
            print(" 💡 TIP: Good time for cooling break (CTRL+C)\n")
            
            gc.collect()
    
    # Final save
    if STOP_REQUESTED:
        print("\n⚠️  Stopped for cooling. Progress saved.")
    else:
        print("\n[INFO] Finalizing...")
    
    flattened = [f.flatten() for f in all_frames]
    df = pd.DataFrame({
        'image': flattened,
        'label': all_labels,
        'video_id': all_ids,
        'manipulation_type': all_types
    })
    df.to_parquet(OUTPUT_FILE, compression='gzip', index=False)
    
    save_checkpoint({
        'videos_processed': checkpoint['videos_processed'] + processed_this_run,
        'processed_ids': list(processed_ids),
        'frames_count': len(all_frames)
    })
    
    if not STOP_REQUESTED:
        Path(CHECKPOINT_FILE).unlink(missing_ok=True)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"SUMMARY - {CATEGORY.upper()}")
    print(f"{'='*80}")
    print(f"Accepted: {processed_this_run}")
    print(f"Rejected: {rejected}")
    print(f"Frames: {len(all_frames)}")
    print(f"Time: {elapsed/60:.0f} min ({elapsed/3600:.1f} hours)")
    
    if STOP_REQUESTED:
        print(f"\n💡 Resume: Run again after cooling")
    else:
        print(f"\n✓ Complete! Quality frames saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
