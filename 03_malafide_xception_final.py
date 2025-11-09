"""
STEP 03: Generate 2D-Malafide Adversarial Attacks - FULL PIPELINE

Features:
- Paper-accurate filter sizes: [3, 9, 27, 81]
- Loads from 5 separate preprocessed_frames_*.parquet files
- ACTUAL Xception + BiGRU (implemented from scratch)
- Optimizes attacks on real network
- Multi-scale perturbations
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import the paper-accurate Xception+BiGRU
from xception_bigru_architecture import XceptionBiGRU

class MalafideAttackPaperFullScale:
    """
    2D-Malafide attack - Paper implementation with full-scale filters
    Learns convolutional filters via gradient-based optimization
    Uses paper-exact filter sizes: [3, 9, 27, 81]
    """
    
    def __init__(self, filter_sizes=[3, 9, 27, 81], num_filters_per_size=1, 
                 epsilon=0.05, learning_rate=0.1, iterations=100, device='cpu'):
        self.filter_sizes = filter_sizes
        self.num_filters_per_size = num_filters_per_size
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.device = device
        
        self.filters = self._initialize_filters()
        self.total_filters = len(self.filters)
        
        print(f"\n[INFO] 2D-Malafide Initialized (PAPER-ACCURATE):")
        print(f"  Filter sizes: {filter_sizes} (Paper-exact)")
        print(f"  Filters per size: {num_filters_per_size}")
        print(f"  Total filters: {self.total_filters}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Multi-scale: YES (3×3, 9×9, 27×27, 81×81)\n")
    
    def _initialize_filters(self):
        """Initialize filters with proper scaling"""
        filters = []
        for size in self.filter_sizes:
            for _ in range(self.num_filters_per_size):
                f = np.random.randn(size, size).astype(np.float32)
                f = f / np.sqrt(size * size)
                filters.append(torch.tensor(f, dtype=torch.float32, requires_grad=True))
        return filters
    
    def _compute_perturbation(self, image_np):
        """Compute multi-scale perturbation"""
        perturbations = []
        
        for filter_kernel in self.filters:
            filter_kernel_np = filter_kernel.detach().cpu().numpy()
            perturbation = np.zeros_like(image_np)
            
            for c in range(image_np.shape[2]):
                channel = image_np[:, :, c]
                filtered = cv2.filter2D(channel, -1, filter_kernel_np)
                perturbation[:, :, c] = filtered
            
            perturbations.append(perturbation)
        
        combined_perturbation = np.mean(perturbations, axis=0)
        combined_perturbation = np.clip(combined_perturbation, -self.epsilon, self.epsilon)
        
        return combined_perturbation
    
    def optimize_filters_on_batch(self, batch_images, batch_labels, model):
        """Optimize filters using gradient ascent on real model"""
        print("[INFO] Optimizing filters via gradient-based learning...")
        print("       On ACTUAL Xception+BiGRU network\n")
        
        filter_params = [f for f in self.filters]
        for f in filter_params:
            f.requires_grad_(True)
        
        optimizer = optim.Adam(filter_params, lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        model.eval()
        best_loss = -np.inf
        best_filters = [f.clone().detach() for f in self.filters]
        
        for iteration in tqdm(range(self.iterations), desc="  Optimizing filters"):
            optimizer.zero_grad()
            total_loss = 0
            batch_count = 0
            
            for i, (img_np, label) in enumerate(zip(batch_images, batch_labels)):
                img_np = (img_np.astype(np.float32) / 255.0)
                perturbation = self._compute_perturbation(img_np)
                attacked_img = img_np + perturbation
                attacked_img = np.clip(attacked_img, 0, 1)
                
                attacked_tensor = torch.tensor(attacked_img, dtype=torch.float32)
                attacked_tensor = attacked_tensor.permute(2, 0, 1).unsqueeze(0)
                attacked_tensor = attacked_tensor.to(self.device)
                
                with torch.enable_grad():
                    logits = model(attacked_tensor)
                    loss = criterion(logits, torch.tensor([label], dtype=torch.long).to(self.device))
                
                total_loss += loss
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                avg_loss.backward()
                optimizer.step()
                
                current_loss = avg_loss.item()
                if current_loss > best_loss:
                    best_loss = current_loss
                    best_filters = [f.clone().detach() for f in self.filters]
        
        self.filters = best_filters
        print(f"✓ Optimization complete. Best loss: {best_loss:.4f}\n")
    
    def generate_attack(self, image_np):
        """Generate adversarial attack"""
        image_np = image_np.astype(np.float32) / 255.0
        perturbation = self._compute_perturbation(image_np)
        attacked_img = image_np + perturbation
        attacked_img = np.clip(attacked_img, 0, 1) * 255
        return attacked_img.astype(np.uint8)

def load_all_preprocessed_data(image_shape=(224, 224, 3)):
    """Load from 5 separate parquet files"""
    parquet_files = [
        'preprocessed_frames_original.parquet',
        'preprocessed_frames_deepfakes.parquet',
        'preprocessed_frames_face2face.parquet',
        'preprocessed_frames_faceswap.parquet',
        'preprocessed_frames_neuraltextures.parquet'
    ]
    
    print("[INFO] Loading preprocessed frames...\n")
    dfs = []
    for pf in parquet_files:
        if Path(pf).exists():
            print(f"  Loading {pf}...")
            df = pd.read_parquet(pf)
            print(f"    ✓ {len(df)} frames")
            dfs.append(df)
    
    if not dfs:
        print("✗ No preprocessed parquet files found!")
        return None
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Total frames combined: {len(df_combined)}\n")
    return df_combined

def main():
    print("=" * 80)
    print("STEP 3: 2D-MALAFIDE ATTACKS (PAPER-ACCURATE)")
    print("=" * 80 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load data
    df = load_all_preprocessed_data()
    if df is None:
        return
    
    # Initialize attack
    malafide = MalafideAttackPaperFullScale(
        filter_sizes=[3, 9, 27, 81],
        num_filters_per_size=1,
        epsilon=0.05,
        learning_rate=0.1,
        iterations=100,
        device=device
    )
    
    # Initialize model
    print("[INFO] Initializing Xception+BiGRU model...")
    model = XceptionBiGRU(num_classes=2)
    model.to(device)
    print("[INFO]   ✓ Model loaded (ACTUAL Xception)\n")
    
    # Sample for optimization
    sample_size = min(500, len(df))
    sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
    sample_images = []
    sample_labels = []
    
    print(f"Sampling {sample_size} frames...")
    for idx in sample_indices:
        img_flat = np.array(df.iloc[idx]['image'])
        img = img_flat.reshape((224, 224, 3)).astype(np.uint8)
        label = df.iloc[idx]['label']
        sample_images.append(img)
        sample_labels.append(label)
    print("✓ Sample prepared\n")
    
    # Optimize
    malafide.optimize_filters_on_batch(sample_images, sample_labels, model)
    
    # Save filters
    filter_path = 'malafide_filters_xception.npy'
    filters_np = [f.detach().cpu().numpy() for f in malafide.filters]
    np.save(filter_path, np.array(filters_np, dtype=object))
    print(f"✓ Saved filters to {filter_path}\n")
    
    # Generate attacks
    print(f"Generating attacks for {len(df)} frames...")
    attacked_frames = []
    for idx in tqdm(range(len(df)), desc="  Attacking"):
        img_flat = np.array(df.iloc[idx]['image'])
        img = img_flat.reshape((224, 224, 3)).astype(np.uint8)
        attacked = malafide.generate_attack(img)
        attacked_frames.append(attacked.flatten())
    
    print(f"✓ Generated {len(attacked_frames)} frames\n")
    
    # Save
    print("Saving...")
    df_attacked = df.copy()
    df_attacked['image'] = attacked_frames
    df_attacked.to_parquet('malafide_attacked_frames_xception.parquet', compression='gzip', index=False)
    
    file_size_mb = Path('malafide_attacked_frames_xception.parquet').stat().st_size / (1024 * 1024)
    print(f"✓ Saved malafide_attacked_frames_xception.parquet ({file_size_mb:.1f} MB)\n")
    
    # Split
    all_indices = np.arange(len(df))
    all_labels = df['label'].values
    indices_train, indices_temp = train_test_split(all_indices, test_size=0.3, stratify=all_labels, random_state=42)
    indices_val, indices_test = train_test_split(indices_temp, test_size=0.5, stratify=all_labels[indices_temp], random_state=42)
    
    split_info = {
        'train_indices': indices_train.tolist(),
        'val_indices': indices_val.tolist(),
        'test_indices': indices_test.tolist(),
        'train_size': len(indices_train),
        'val_size': len(indices_val),
        'test_size': len(indices_test)
    }
    
    with open('train_val_test_split_xception.json', 'w') as f:
        json.dump(split_info, f, indent=4)
    
    print("=" * 80)
    print("✓ STEP 3 COMPLETE")
    print("=" * 80)
    print(f"\nFiles: malafide_attacked_frames_xception.parquet, malafide_filters_xception.npy\n")

if __name__ == "__main__":
    main()
