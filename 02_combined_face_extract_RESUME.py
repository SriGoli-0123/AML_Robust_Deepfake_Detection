"""
STEP 2: Combined Face Video + Frame Extraction with Stop/Resume
Combines 02a+02b into one script with CTRL+C safe checkpointing

Process:
1. Create face-tracked video (256×256, 30% margin, smoothed boxes)
2. Extract 10 frames from face video (224×224, enhanced)
3. Save checkpoint every video
4. CTRL+C safe - resume anytime

Quality: Same as 02a+02b (face-tracked stability)
Speed: Still slow but resumable with progress tracking
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
import tempfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configuration
FRAMES_PER_VIDEO = 10
TARGET_FACE_SIZE = 224
FACE_MARGIN = 0.3  # 30% margin (better quality)
MIN_FACE_FRAMES = 50
SMOOTHING_WINDOW = 5
FACE_VIDEO_SIZE = 256
CHECKPOINT_INTERVAL = 1  # Save after EVERY video (for stop/resume)

CATEGORY = "original"  # CHANGE: original, deepfakes, face2face, faceswap, neuraltextures

OUTPUT_FILE = f"preprocessed_frames_{CATEGORY}.parquet"
CHECKPOINT_FILE = f"checkpoint_{CATEGORY}.json"
TEMP_DIR = Path(f"temp_face_videos_{CATEGORY}")

# Global flag for graceful shutdown
STOP_REQUESTED = False


def signal_handler(sig, frame):
    """Handle CTRL+C gracefully"""
    global STOP_REQUESTED
    print('\n\n⚠️  Stop requested! Finishing current video and saving...')
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, signal_handler)


class CombinedExtractor:
    """Combined face video creation + frame extraction"""
    
    def __init__(self):
        self.mtcnn = MTCNN()
        self.dlib_detector = dlib.get_frontal_face_detector()
        
    def detect_face_box(self, frame):
        """Detect largest face in frame"""
        try:
            faces = self.mtcnn.detect_faces(frame)
            if faces:
                largest = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                x, y, w, h = largest['box']
                return (x, y, w, h), largest.get('confidence', 1.0)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = self.dlib_detector(gray, 1)
            if len(dets) > 0:
                d = dets[0]
                return (d.left(), d.top(), d.width(), d.height()), 0.9
            
            return None, 0.0
        except:
            return None, 0.0
    
    def smooth_boxes(self, boxes, window=SMOOTHING_WINDOW):
        """Smooth bounding boxes"""
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
        """Crop frame to face with 30% margin"""
        h, w = frame.shape[:2]
        x, y, fw, fh = face_box
        
        # 30% margin
        margin_w = int(fw * FACE_MARGIN)
        margin_h = int(fh * FACE_MARGIN)
        
        # Square crop
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
    
    def create_face_video_memory(self, video_path):
        """Create face video in memory (no disk write)"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # PASS 1: Detect faces
            face_boxes = []
            confidences = []
            frames_buffer = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames_buffer.append(frame)
                box, conf = self.detect_face_box(frame)
                
                if box:
                    face_boxes.append(box)
                    confidences.append(conf)
                else:
                    if len(face_boxes) > 0:
                        face_boxes.append(face_boxes[-1])
                        confidences.append(0.5)
                    else:
                        face_boxes.append((0, 0, frame.shape[1], frame.shape[0]))
                        confidences.append(0.1)
            
            cap.release()
            
            if len(face_boxes) < MIN_FACE_FRAMES:
                return None, 0.0
            
            # PASS 2: Smooth boxes
            smoothed_boxes = self.smooth_boxes(face_boxes)
            
            # PASS 3: Create face frames in memory
            face_frames = []
            for frame, box in zip(frames_buffer, smoothed_boxes):
                crop = self.crop_frame_to_face(frame, box)
                if crop is not None:
                    face_frames.append(crop)
            
            avg_conf = np.mean(confidences)
            return face_frames, avg_conf
            
        except Exception as e:
            return None, 0.0
        finally:
            if 'cap' in locals():
                cap.release()
    
    def extract_frames_from_face_video(self, face_frames, label, manip_type, video_id, category_suffix):
        """Extract 10 frames from face video frames"""
        if not face_frames or len(face_frames) < FRAMES_PER_VIDEO:
            return []
        
        # Select 10 evenly-spaced frames
        total = len(face_frames)
        interval = total // (FRAMES_PER_VIDEO + 1)
        
        frames = []
        video_id_full = f"{video_id}_{category_suffix}"
        
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
            
            frames.append({
                'image': face_enhanced,
                'label': label,
                'video_id': video_id_full,
                'manipulation_type': manip_type
            })
        
        return frames
    
    def process_video(self, video_path, label, manip_type):
        """Combined: Create face video + extract frames"""
        # Step 1: Create face video in memory
        face_frames, conf = self.create_face_video_memory(video_path)
        
        if face_frames is None:
            return [], 0.0
        
        # Step 2: Extract 10 frames
        frames = self.extract_frames_from_face_video(
            face_frames,
            label,
            manip_type,
            video_path.stem,
            CATEGORY
        )
        
        return frames, conf


def visualize_checkpoint(frames_list, checkpoint_num, category):
    """Visualize sample faces"""
    if not frames_list or len(frames_list) < 10:
        return
    
    num_samples = min(10, len(frames_list))
    sample_indices = np.random.choice(len(frames_list), num_samples, replace=False)
    samples = [frames_list[i] for i in sample_indices]
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.3)
    
    for i in range(min(10, len(samples))):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        face = samples[i]
        
        if len(face.shape) == 1:
            face = face.reshape(TARGET_FACE_SIZE, TARGET_FACE_SIZE, 3)
        
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        ax.imshow(face_rgb)
        ax.set_title(f"Sample {i+1}", fontsize=9, fontweight='bold')
        ax.axis('off')
    
    ax_hist = fig.add_subplot(gs[2, :2])
    sample_face = samples[0]
    if len(sample_face.shape) == 1:
        sample_face = sample_face.reshape(TARGET_FACE_SIZE, TARGET_FACE_SIZE, 3)
    
    for channel_idx, color, label in [(0, 'b', 'Blue'), (1, 'g', 'Green'), (2, 'r', 'Red')]:
        hist = cv2.calcHist([sample_face], [channel_idx], None, [256], [0, 256])
        ax_hist.plot(hist, color=color.lower(), label=label, alpha=0.7, linewidth=2)
    
    ax_hist.set_title("RGB Distribution", fontsize=10, fontweight='bold')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(alpha=0.3)
    
    ax_stats = fig.add_subplot(gs[2, 2:])
    ax_stats.axis('off')
    
    b_mean = sample_face[:,:,0].mean()
    g_mean = sample_face[:,:,1].mean()
    r_mean = sample_face[:,:,2].mean()
    overall_mean = sample_face.mean()
    contrast = sample_face.std()
    
    is_color = not (abs(b_mean - g_mean) < 5 and abs(g_mean - r_mean) < 5)
    color_status = "✓ TRUE RGB" if is_color else "✗ GRAYSCALE"
    
    stats_text = f"""FACE-TRACKED EXTRACTION
━━━━━━━━━━━━━━━━━━━━━
Category: {category.upper()}
Checkpoint: #{checkpoint_num}
Frames: {len(frames_list)}

COLOR: {color_status}
Brightness: {overall_mean:.1f}
Contrast: {contrast:.1f}

✓ Face-tracked (smooth)
✓ 30% margin → 256×256 → 224×224
"""
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=9, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    output_file = f"preprocessing_sample_{category}.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Visualization: {output_file}")


def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'videos_processed': 0, 'processed_ids': [], 'frames_count': 0, 'visualized': False}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def load_config():
    with open('config_scaled.json') as f:
        cfg = json.load(f)
    
    path_patterns = cfg['categorized_video_paths'][CATEGORY]
    videos = []
    for pattern in path_patterns:
        videos.extend(glob.glob(pattern))
    
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
    print(f"COMBINED FACE-TRACKED EXTRACTION - {CATEGORY.upper()}")
    print("=" * 80)
    print(f"[INFO] Process: Face tracking (256×256) → 10 frames (224×224)")
    print(f"[INFO] Margin: 30%, Smoothing: {SMOOTHING_WINDOW} frames")
    print(f"[INFO] Press CTRL+C to stop safely anytime\n")
    
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint['processed_ids'])
    visualized = checkpoint.get('visualized', False)
    
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
    
    print(f"[INFO] Total: {len(all_videos)}")
    print(f"[INFO] Done: {len(processed_ids)}")
    print(f"[INFO] Remaining: {len(remaining)}\n")
    
    if len(remaining) == 0:
        print("✓ All processed!")
        return
    
    extractor = CombinedExtractor()
    start_time = time.time()
    processed_this_run = 0
    
    for idx, (video_path, label, manip_type) in enumerate(remaining):
        if STOP_REQUESTED:
            break
        
        print(f"[{idx+1}/{len(remaining)}] {video_path.name}... ", end='', flush=True)
        
        frames, conf = extractor.process_video(video_path, label, manip_type)
        
        if frames:
            for fd in frames:
                all_frames.append(fd['image'])
                all_labels.append(fd['label'])
                all_ids.append(fd['video_id'])
                all_types.append(fd['manipulation_type'])
            print(f"✓ {len(frames)} frames (conf={conf:.2f})")
        else:
            print("✗ Failed")
        
        processed_ids.add(str(video_path))
        processed_this_run += 1
        
        # Save checkpoint after EVERY video
        elapsed = time.time() - start_time
        rate = processed_this_run / elapsed
        eta = (len(remaining) - processed_this_run) / rate if rate > 0 else 0
        
        # Show progress every 10 videos
        if processed_this_run % 10 == 0:
            print(f"\n[PROGRESS] {processed_this_run}/{len(remaining)} videos")
            print(f" Speed: {rate*60:.1f} videos/hour")
            print(f" ETA: {eta/3600:.1f} hours")
            print(f" Frames: {len(all_frames)}")
            
            if not visualized and len(all_frames) >= 20:
                print(" 📊 Creating visualization...")
                visualize_checkpoint(all_frames, processed_this_run, CATEGORY)
                visualized = True
            print()
        
        # Save checkpoint
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
            'frames_count': len(all_frames),
            'visualized': visualized
        })
        
        gc.collect()
    
    if STOP_REQUESTED:
        print("\n⚠️  Stopped. Progress saved.")
        print("💡 Resume: Run script again")
    else:
        Path(CHECKPOINT_FILE).unlink(missing_ok=True)
        print("\n✓ Complete!")
    
    elapsed = time.time() - start_time
    print(f"\nFrames: {len(all_frames)}")
    print(f"Videos: {processed_this_run}")
    print(f"Time: {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
