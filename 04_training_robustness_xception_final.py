"""
STEP 04: Train 5 Models + Robustness Evaluation - XCEPTION+BiGRU

Models:
1. Baseline (clean)
2. FGSM-Robust
3. PGD-Robust
4. Malafide-Robust
5. Adversarial-Robust (all attacks combined)

Uses ACTUAL Xception+BiGRU throughout
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from xception_bigru_architecture import XceptionBiGRU

def load_dataset(parquet_file):
    if not Path(parquet_file).exists():
        return None
    df = pd.read_parquet(parquet_file)
    images = [np.array(img).reshape((224, 224, 3)) for img in df['image']]
    X = torch.tensor(np.array(images), dtype=torch.float32) / 255.0
    X = X.permute(0, 3, 1, 2)
    y = torch.tensor(df['label'].values, dtype=torch.long)
    return X, y

def prepare_loaders(X, y, batch_size=32):
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, epochs=5, device='cpu', model_name='model'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
        print(f"Val Acc: {val_correct/val_total:.2%}")
    
    return model

def test_model(model, test_loaders_dict, device='cpu'):
    model.eval()
    results = {}
    with torch.no_grad():
        for attack_type, test_loader in test_loaders_dict.items():
            correct, total = 0, 0
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
            results[attack_type] = correct / total if total > 0 else 0
    return results

def main():
    print("=" * 80)
    print("STEP 4: TRAIN 5 MODELS + ROBUSTNESS EVALUATION")
    print("=" * 80)
    print("Using ACTUAL Xception+BiGRU\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load datasets
    print("Loading datasets...\n")
    X_clean, y_clean = load_dataset('attacked_frames_clean.parquet')
    X_fgsm, y_fgsm = load_dataset('attacked_frames_fgsm.parquet')
    X_pgd, y_pgd = load_dataset('attacked_frames_pgd.parquet')
    X_malafide, y_malafide = load_dataset('malafide_attacked_frames_xception.parquet')
    
    if X_clean is None:
        print("✗ No datasets found")
        return
    
    print("✓ All datasets loaded\n")
    
    # Prepare loaders
    train_clean, val_clean, test_clean = prepare_loaders(X_clean, y_clean)
    _, _, test_fgsm = prepare_loaders(X_fgsm, y_fgsm)
    _, _, test_pgd = prepare_loaders(X_pgd, y_pgd)
    _, _, test_malafide = prepare_loaders(X_malafide, y_malafide)
    
    test_loaders = {
        'Clean': test_clean,
        'FGSM': test_fgsm,
        'PGD': test_pgd,
        'Malafide': test_malafide
    }
    
    all_results = {}
    
    # Model 1: Baseline
    print("[1/5] Baseline (Clean)")
    model_clean = XceptionBiGRU(num_classes=2)
    model_clean = train_model(model_clean, train_clean, val_clean, epochs=5, device=device)
    all_results['Baseline (Clean)'] = test_model(model_clean, test_loaders, device)
    print(f"Results: {all_results['Baseline (Clean)']}\n")
    
    # Model 2: FGSM
    print("[2/5] FGSM-Robust")
    train_fgsm, val_fgsm, _ = prepare_loaders(X_fgsm, y_fgsm)
    model_fgsm = XceptionBiGRU(num_classes=2)
    model_fgsm = train_model(model_fgsm, train_fgsm, val_fgsm, epochs=5, device=device)
    all_results['FGSM-Robust'] = test_model(model_fgsm, test_loaders, device)
    print(f"Results: {all_results['FGSM-Robust']}\n")
    
    # Model 3: PGD
    print("[3/5] PGD-Robust")
    train_pgd, val_pgd, _ = prepare_loaders(X_pgd, y_pgd)
    model_pgd = XceptionBiGRU(num_classes=2)
    model_pgd = train_model(model_pgd, train_pgd, val_pgd, epochs=5, device=device)
    all_results['PGD-Robust'] = test_model(model_pgd, test_loaders, device)
    print(f"Results: {all_results['PGD-Robust']}\n")
    
    # Model 4: Malafide
    print("[4/5] Malafide-Robust")
    train_malafide, val_malafide, _ = prepare_loaders(X_malafide, y_malafide)
    model_malafide = XceptionBiGRU(num_classes=2)
    model_malafide = train_model(model_malafide, train_malafide, val_malafide, epochs=5, device=device)
    all_results['Malafide-Robust'] = test_model(model_malafide, test_loaders, device)
    print(f"Results: {all_results['Malafide-Robust']}\n")
    
    # Model 5: Combined
    print("[5/5] Adversarial-Robust (All Combined)")
    X_combined = torch.cat([X_fgsm, X_pgd, X_malafide], dim=0)
    y_combined = torch.cat([y_fgsm, y_pgd, y_malafide], dim=0)
    train_combined, val_combined, _ = prepare_loaders(X_combined, y_combined)
    model_combined = XceptionBiGRU(num_classes=2)
    model_combined = train_model(model_combined, train_combined, val_combined, epochs=5, device=device)
    all_results['Adversarial-Robust'] = test_model(model_combined, test_loaders, device)
    print(f"Results: {all_results['Adversarial-Robust']}\n")
    
    # Save
    with open('robustness_evaluation_results_xception.json', 'w') as f:
        results_serializable = {model_name: {k: float(v) for k, v in metrics.items()} 
                               for model_name, metrics in all_results.items()}
        json.dump(results_serializable, f, indent=4)
    
    print("=" * 80)
    print("✓ STEP 4 COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to robustness_evaluation_results_xception.json\n")

if __name__ == "__main__":
    main()
