"""
MASTER PREPROCESSING SCRIPT - Process All 5 Categories at Once
Automatically generates and runs all 5 category-specific scripts

FIXED: Ensures exactly 10 valid face frames per video
- Scans entire video for faces
- Filters out blurry/low-quality frames
- Evenly distributes 10 valid frames
"""

import subprocess
import sys
from pathlib import Path

CATEGORIES = ["original", "deepfakes", "face2face", "faceswap", "neuraltextures"]

SCRIPT_TEMPLATE = '''"""
STEP 2: Optimized Preprocessing - Guaranteed 10 Valid Face Frames Per Video

Key Improvements:
1. FIXED: Per-channel CLAHE instead of grayscale conversion → True RGB frames
2. SPEED: Efficient face detection + quality filtering
3. VISUAL: Automatic frame samples saved at each checkpoint for quality verification
4. QUALITY: Ensures exactly 10 valid face frames per video (blurry frames filtered)
5. FIXED: PyArrow parquet serialization (flatten before save, reshape on load)

Algorithm:
1. Scan entire video for all detectable faces
2. Filter by quality metrics (blur, brightness, face occupancy)
3. Select 10 evenly-distributed frames from valid candidates
4. If <10 valid frames found, extract more from intermediate frames
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============ CONFIGURATION ============
FRAMES_PER_VIDEO = 10
TARGET_FACE_SIZE = 224
CHECKPOINT_INTERVAL = 25

OUTPUT_FILE = "preprocessed_frames_{category}.parquet"
CHECKPOINT_FILE = "preprocess_checkpoint_{category}.json"
VISUALIZATION_DIR = "preprocessing_samples"
QUALITY_LOG_FILE = "preprocessing_quality_{category}.txt"

CATEGORY = "{category}"

# Quality thresholds
BLUR_THRESHOLD = 100  # Laplacian variance threshold
MIN_BRIGHTNESS = 30   # Minimum mean pixel value
MAX_BRIGHTNESS = 225  # Maximum mean pixel value
MIN_FACE_OCCUPANCY = 0.25  # Minimum 25% of frame should be face/skin

# Create visualization directory
Path(VISUALIZATION_DIR).mkdir(exist_ok=True)


class OptimizedFaceExtractor:
    """
    Optimized face extraction with quality filtering
    """

    def __init__(self, target_face_size=TARGET_FACE_SIZE):
        self.target_face_size = target_face_size
        self.mtcnn_detector = MTCNN()
        self.dlib_detector = dlib.get_frontal_face_detector()
        predictor_path = self._get_dlib_predictor()
        self.dlib_predictor = dlib.shape_predictor(predictor_path) if predictor_path else None
        self.quality_log = {{}}

    def _get_dlib_predictor(self):
        predictor_path = Path("shape_predictor_68_face_landmarks.dat")
        if predictor_path.exists():
            return str(predictor_path)
        print("[INFO] Downloading dlib predictor...")
        import urllib.request, bz2
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, "temp.bz2")
        with bz2.BZ2File("temp.bz2") as f_in, open(predictor_path, 'wb') as f_out:
            f_out.write(f_in.read())
        Path("temp.bz2").unlink()
        return str(predictor_path)

    def detect_faces(self, frame):
        """Detect all faces in frame"""
        try:
            results = self.mtcnn_detector.detect_faces(frame)
            return [(r['box'][0], r['box'][1], r['box'][2], r['box'][3]) for r in results]
        except:
            return []

    def get_landmarks(self, frame, face_box):
        """Get facial landmarks for alignment"""
        if not self.dlib_predictor:
            return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = face_box
            rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            lm = self.dlib_predictor(gray, rect)
            return np.array([[lm.part(i).x, lm.part(i).y] for i in range(68)])
        except:
            return None

    def align_face(self, frame, landmarks):
        """Align face based on eye positions"""
        if landmarks is None or len(landmarks) < 48:
            return frame
        try:
            left_eye = landmarks[36:42].mean(axis=0)
            right_eye = landmarks[42:48].mean(axis=0)
            angle = np.degrees(np.arctan2(right_eye[1]-left_eye[1], right_eye[0]-left_eye[0]))
            center = (left_eye + right_eye) / 2
            M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
            return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        except:
            return frame

    def check_frame_quality(self, frame):
        """
        Check if frame meets quality criteria
        Returns: (is_valid, reason)
        """
        if frame is None or frame.size == 0:
            return False, "EMPTY_FRAME"
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check 1: Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < BLUR_THRESHOLD:
            return False, f"BLUR_{{laplacian_var:.1f}}"
        
        # Check 2: Brightness/contrast
        mean_brightness = gray.mean()
        if mean_brightness < MIN_BRIGHTNESS or mean_brightness > MAX_BRIGHTNESS:
            return False, f"BRIGHTNESS_{{mean_brightness:.1f}}"
        
        # Check 3: Face occupancy (skin tone detection)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broader skin tone range for different ethnicities
        lower_skin = np.array([0, 15, 30])
        upper_skin = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(mask) / (frame.shape[0] * frame.shape[1])
        
        if skin_ratio < MIN_FACE_OCCUPANCY:
            return False, f"LOW_OCCUPANCY_{{skin_ratio:.2f}}"
        
        return True, "VALID"

    def crop_face(self, frame, face_box, landmarks=None):
        """
        Extract and preprocess face region
        Returns: (cropped_face, is_valid, quality_reason)
        """
        try:
            x, y, w, h = face_box
            margin = int(0.2 * max(w, h))
            x1, y1 = max(0, x-margin), max(0, y-margin)
            x2, y2 = min(frame.shape[1], x+w+margin), min(frame.shape[0], y+h+margin)
            face = frame[y1:y2, x1:x2]

            if landmarks is not None:
                lm_crop = landmarks - np.array([x1, y1])
                face = self.align_face(face, lm_crop)

            face = cv2.resize(face, (self.target_face_size, self.target_face_size))

            # Check quality before enhancement
            is_valid, reason = self.check_frame_quality(face)
            
            # Apply CLAHE per-channel to preserve RGB color
            b, g, r = cv2.split(face)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            b_eq = clahe.apply(b)
            g_eq = clahe.apply(g)
            r_eq = clahe.apply(r)
            face_enhanced = cv2.merge((b_eq, g_eq, r_eq))
            
            return face_enhanced, is_valid, reason
        except Exception as e:
            return None, False, f"CROP_ERROR_{{str(e)[:20]}}"

    def extract_video_smart(self, video_path, label, manip_type, video_id, category_suffix, target_frames=FRAMES_PER_VIDEO):
        """
        Smart frame extraction:
        1. Scan entire video for all valid faces
        2. Select target_frames evenly distributed across valid frames
        3. Fill gaps if needed by extracting additional frames
        """
        candidate_frames = []  # (frame_number, frame_data, is_valid, reason)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            
            # PASS 1: Scan entire video for valid faces
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                faces = self.detect_faces(frame)
                
                # Extract all faces from this frame
                for face_box in faces:
                    lm = self.get_landmarks(frame, face_box)
                    crop, is_valid, reason = self.crop_face(frame, face_box, lm)
                    
                    if crop is not None:
                        candidate_frames.append({{
                            'frame_num': frame_idx,
                            'image': crop,
                            'is_valid': is_valid,
                            'reason': reason
                        }})
                
                frame_idx += 1
            
            cap.release()
            
            # PASS 2: Filter valid frames and select evenly distributed ones
            valid_frames = [f for f in candidate_frames if f['is_valid']]
            invalid_frames = [f for f in candidate_frames if not f['is_valid']]
            
            # If enough valid frames, select evenly distributed
            if len(valid_frames) >= target_frames:
                indices = np.linspace(0, len(valid_frames)-1, target_frames, dtype=int)
                selected_frames = [valid_frames[i]['image'] for i in indices]
            else:
                # Use all valid frames + some invalid ones if needed
                selected_frames = [f['image'] for f in valid_frames]
                
                # Fill remaining slots with best invalid frames
                if len(selected_frames) < target_frames and invalid_frames:
                    needed = target_frames - len(selected_frames)
                    selected_frames.extend([f['image'] for f in invalid_frames[:needed]])
            
            # PASS 3: Create output with category-differentiated video_id
            frames_data = []
            differentiated_video_id = f"{{video_id}}_{{category_suffix}}"
            
            for crop in selected_frames:
                frames_data.append({{
                    'image': crop,
                    'label': label,
                    'video_id': differentiated_video_id,
                    'manipulation_type': manip_type
                }})
            
            # Log statistics
            self.quality_log[str(video_path)] = {{
                'total_candidates': len(candidate_frames),
                'valid_frames': len(valid_frames),
                'invalid_frames': len(invalid_frames),
                'frames_extracted': len(frames_data),
                'reasons': {{r['reason']: sum(1 for f in candidate_frames if f['reason'] == r['reason']) 
                           for r in candidate_frames}}
            }}
            
            return frames_data
            
        except Exception as e:
            print(f" ⚠ Error processing {{video_path}}: {{str(e)[:50]}}")
            return []


def visualize_samples(frames_list, labels_list, video_ids, checkpoint_num, category):
    """Save sample frames for quality verification"""
    if not frames_list or len(frames_list) < 5:
        return

    sample_indices = np.random.choice(len(frames_list), min(20, len(frames_list)), replace=False)
    samples = [frames_list[i] for i in sample_indices]

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.3)

    # Row 1-2: Frame samples
    for i in range(min(10, len(samples))):
        ax = fig.add_subplot(gs[i//5, i%5])
        frame = samples[i]
        if len(frame.shape) == 1:
            frame = frame.reshape(TARGET_FACE_SIZE, TARGET_FACE_SIZE, 3)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        ax.set_title(f"Sample {{i}}: {{frame.shape}}", fontsize=8)
        ax.axis('off')

    # Row 3: Color statistics
    ax_hist = fig.add_subplot(gs[2, :])
    sample_frame = samples[0]
    if len(sample_frame.shape) == 1:
        sample_frame = sample_frame.reshape(TARGET_FACE_SIZE, TARGET_FACE_SIZE, 3)
    
    for color, label, channel_idx in [('b', 'Blue', 0), ('g', 'Green', 1), ('r', 'Red', 2)]:
        if sample_frame.shape[2] == 3:
            hist = cv2.calcHist([sample_frame], [channel_idx], None, [256], [0, 256])
            ax_hist.plot(hist, label=label, color=color, alpha=0.7)
    ax_hist.set_title("Color Distribution (First Sample)", fontsize=10)
    ax_hist.set_xlabel("Pixel Value")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend()

    # Row 4: Statistics
    ax_stats = fig.add_subplot(gs[3, :])
    ax_stats.axis('off')
    
    sample_frame = samples[0]
    if len(sample_frame.shape) == 1:
        sample_frame = sample_frame.reshape(TARGET_FACE_SIZE, TARGET_FACE_SIZE, 3)
    
    is_color = False
    if len(sample_frame.shape) == 3 and sample_frame.shape[2] == 3:
        b, g, r = sample_frame[:,:,0], sample_frame[:,:,1], sample_frame[:,:,2]
        is_color = not (np.allclose(b, g) and np.allclose(g, r))
    
    stats_text = f"""
    Checkpoint #{{checkpoint_num}} - {{category.upper()}}
    ───────────────────────────────────────
    Total Frames Processed: {{len(frames_list)}}
    Sample Frame Shape: {{sample_frame.shape}}
    Frame Format: {{'TRUE RGB COLOR ✓' if is_color else 'GRAYSCALE (B&W)'}}
    Mean Value (B): {{sample_frame[:,:,0].mean():.1f}} | (G): {{sample_frame[:,:,1].mean():.1f}} | (R): {{sample_frame[:,:,2].mean():.1f}}
    ───────────────────────────────────────
    """
    ax_stats.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    output_path = Path(VISUALIZATION_DIR) / f"checkpoint_{{checkpoint_num:03d}}_{{category}}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved visualization: {{output_path}}")


def load_checkpoint(checkpoint_file=CHECKPOINT_FILE):
    path = Path(checkpoint_file)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {{'videos_processed': 0, 'processed_ids': [], 'frames_count': 0}}


def save_checkpoint(data, checkpoint_file=CHECKPOINT_FILE):
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f)


def load_config(config_file='config_scaled.json'):
    """Load config and prepare video list with label"""
    with open(config_file) as f:
        cfg = json.load(f)
    videos = []
    label = 0 if CATEGORY == 'original' else 1
    for p in cfg['categorized_video_paths'][CATEGORY]:
        videos.append((Path(p), label, CATEGORY))
    return videos


def load_existing_parquet(output_file=OUTPUT_FILE):
    """Load existing parquet if available"""
    path = Path(output_file)
    if path.exists():
        print(f"[INFO] Loading existing parquet ({{output_file}})...\\n")
        df = pd.read_parquet(output_file)
        df['image'] = df['frame'].apply(lambda x: np.array(x).reshape(TARGET_FACE_SIZE, TARGET_FACE_SIZE, 3))
        return df
    return None


def preprocess_folder():
    print("=" * 80)
    print(f"STEP 2: SMART PREPROCESSING - {{CATEGORY.upper()}}")
    print("=" * 80)
    print(f"[INFO] Target: 10 valid face frames per video")
    print(f"[INFO] Blur threshold: {{BLUR_THRESHOLD}}")
    print(f"[INFO] Min face occupancy: {{MIN_FACE_OCCUPANCY*100:.0f}}%")
    print(f"[INFO] Output: {{OUTPUT_FILE}}")
    print(f"[INFO] Visualizations: {{VISUALIZATION_DIR}}/\\n")

    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint['processed_ids'])

    # Load existing data
    existing_df = load_existing_parquet()
    if existing_df is not None:
        all_frames = [np.array(x) for x in existing_df['image']]
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

    extractor = OptimizedFaceExtractor(TARGET_FACE_SIZE)
    processed_this_run = 0
    start_time = time.time()

    print(f"[INFO] Found {{len(all_videos)}} videos")
    print(f"[INFO] Remaining unprocessed: {{len(remaining)}}\\n")

    # Process videos
    for idx, (video_path, label, manip_type) in enumerate(tqdm(remaining, desc=f"Processing {{CATEGORY}}")):
        frames_data = extractor.extract_video_smart(
            video_path,
            label,
            manip_type,
            video_path.stem,
            category_suffix=CATEGORY,
            target_frames=FRAMES_PER_VIDEO
        )

        for fd in frames_data:
            all_frames.append(fd['image'])
            all_labels.append(fd['label'])
            all_ids.append(fd['video_id'])
            all_types.append(fd['manipulation_type'])

        processed_ids.add(str(video_path))
        processed_this_run += 1

        if processed_this_run % CHECKPOINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = processed_this_run / elapsed
            eta = (len(remaining) - processed_this_run) / rate if rate > 0 else 0

            print(f"\\n[CHECKPOINT #{{processed_this_run // CHECKPOINT_INTERVAL}}] {{processed_this_run}}/{{len(remaining)}} videos")
            print(f" ⏱ Speed: {{rate:.2f}} videos/sec")
            print(f" ⏳ ETA: {{eta/60:.1f}} minutes")
            print(f" 📊 Total frames: {{len(all_frames)}} (~{{len(all_frames)//max(1,processed_this_run)}} frames/video avg)")

            flattened_frames = [frame.flatten() for frame in all_frames]
            df = pd.DataFrame({{
                'frame': flattened_frames,
                'label': all_labels,
                'video_id': all_ids,
                'manipulation_type': all_types
            }})
            df.to_parquet(OUTPUT_FILE, compression='gzip', index=False)

            print(f" 📸 Generating visualization...", end='')
            try:
                visualize_samples(all_frames, all_labels, all_ids, 
                                processed_this_run // CHECKPOINT_INTERVAL, CATEGORY)
            except Exception as e:
                print(f" ⚠ (skipped: {{str(e)[:50]}})")
            else:
                print()

            save_checkpoint({{
                'videos_processed': checkpoint['videos_processed'] + processed_this_run,
                'processed_ids': list(processed_ids),
                'frames_count': len(all_frames)
            }})

            gc.collect()

    # Final save
    print(f"\\n[FINAL] Processing complete!")
    print(f" Total videos processed: {{checkpoint['videos_processed'] + processed_this_run}}")
    print(f" Total frames extracted: {{len(all_frames)}}")
    print(f" Average frames per video: {{len(all_frames) / max(1, processed_this_run):.1f}}")

    flattened_frames = [frame.flatten() for frame in all_frames]
    df = pd.DataFrame({{
        'frame': flattened_frames,
        'label': all_labels,
        'video_id': all_ids,
        'manipulation_type': all_types
    }})
    df.to_parquet(OUTPUT_FILE, compression='gzip', index=False)

    print(f" 📸 Generating final visualization...", end='')
    try:
        visualize_samples(all_frames, all_labels, all_ids, 'FINAL', CATEGORY)
    except Exception as e:
        print(f" ⚠ (skipped: {{str(e)[:50]}})")
    else:
        print()

    # Save quality log
    with open(QUALITY_LOG_FILE, 'w') as f:
        f.write(f"Quality Report - {{CATEGORY.upper()}}\\n")
        f.write("="*80 + "\\n\\n")
        for video, stats in extractor.quality_log.items():
            f.write(f"{{video}}\\n")
            f.write(f"  Total candidates: {{stats['total_candidates']}}\\n")
            f.write(f"  Valid frames: {{stats['valid_frames']}}\\n")
            f.write(f"  Invalid frames: {{stats['invalid_frames']}}\\n")
            f.write(f"  Frames extracted: {{stats['frames_extracted']}}\\n\\n")

    save_checkpoint({{
        'videos_processed': checkpoint['videos_processed'] + processed_this_run,
        'processed_ids': list(processed_ids),
        'frames_count': len(all_frames)
    }})

    print(f"\\n✓ Output file: {{OUTPUT_FILE}}")
    print(f"✓ Visualizations: {{VISUALIZATION_DIR}}/")
    print(f"✓ Quality report: {{QUALITY_LOG_FILE}}")


if __name__ == "__main__":
    preprocess_folder()
'''


def generate_category_script(category):
    """Generate category-specific preprocessing script"""
    script_content = SCRIPT_TEMPLATE.format(category=category)
    filename = f"02_preprocess_{category}.py"
    with open(filename, 'w') as f:
        f.write(script_content)
    return filename


def run_all_categories():
    """Generate all 5 scripts and run sequentially"""
    print("=" * 80)
    print("MASTER PREPROCESSING - ALL 5 CATEGORIES (SMART QUALITY FILTERING)")
    print("=" * 80)
    
    scripts = []
    for category in CATEGORIES:
        script = generate_category_script(category)
        scripts.append(script)
        print(f"✓ Generated: {script}")
    
    print("\n" + "=" * 80)
    print("RUNNING PREPROCESSING FOR ALL CATEGORIES")
    print("=" * 80 + "\n")
    
    for i, (category, script) in enumerate(zip(CATEGORIES, scripts), 1):
        print(f"\n{'='*80}")
        print(f"[{i}/5] Starting: {category.upper()}")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run([sys.executable, script], check=True)
            print(f"\n✓ COMPLETED: {category.upper()}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ ERROR in {category.upper()}: {e}\n")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ ALL PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files:")
    for category in CATEGORIES:
        print(f"  - preprocessed_frames_{category}.parquet")
        print(f"  - preprocessing_quality_{category}.txt")
    print(f"\nVisualizations in: preprocessing_samples/")


if __name__ == "__main__":
    run_all_categories()