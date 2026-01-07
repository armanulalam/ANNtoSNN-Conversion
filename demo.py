"""
Quick Demo: Test the ANN-SNN pipeline with a small subset
This script allows you to quickly verify the installation and workflow.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models.ann_cnn import ANN_CNN
from models.snn_cnn import SNN_CNN
from utils import get_cifar10_loaders


def quick_demo():
    """
    Quick demonstration of the ANN-SNN conversion pipeline.
    Uses a small subset of data for fast testing.
    """
    
    print("\n" + "="*70)
    print("Quick Demo: ANN-SNN Conversion Pipeline")
    print("="*70 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Step 1: Load a small batch of data
    print("1. Loading CIFAR-10 dataset...")
    _, test_loader, classes = get_cifar10_loaders(batch_size=32)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    print(f"   ✓ Loaded batch: {images.shape}\n")
    
    # Step 2: Test ANN model
    print("2. Testing ANN CNN model...")
    ann_model = ANN_CNN(num_classes=10).to(device)
    ann_model.eval()
    
    with torch.no_grad():
        ann_output = ann_model(images)
        ann_pred = torch.argmax(ann_output, dim=1)
    
    print(f"   ✓ ANN output shape: {ann_output.shape}")
    print(f"   ✓ Parameters: {ann_model.get_num_parameters():,}\n")
    
    # Step 3: Test SNN model
    print("3. Testing SNN CNN model...")
    snn_model = SNN_CNN(num_classes=10, beta=0.95, num_steps=25).to(device)
    
    # Transfer weights
    snn_model.load_ann_weights(ann_model)
    snn_model.eval()
    
    with torch.no_grad():
        spk_rec, mem_rec, spike_counts = snn_model(images)
        snn_output = spk_rec.sum(dim=0)
        snn_pred = torch.argmax(snn_output, dim=1)
    
    print(f"   ✓ SNN spike recordings: {spk_rec.shape}")
    print(f"   ✓ Total spikes: {sum(spike_counts.values()):,}")
    print(f"   ✓ Spike counts per layer:")
    for layer, count in spike_counts.items():
        print(f"      {layer}: {count:,}")
    print()
    
    # Step 4: Compare predictions
    print("4. Comparing ANN vs SNN predictions...")
    agreement = (ann_pred == snn_pred).float().mean().item() * 100
    print(f"   ✓ Prediction agreement: {agreement:.2f}%")
    
    # Show some example predictions
    print(f"\n   Example predictions (first 5 samples):")
    for i in range(min(5, len(labels))):
        true_label = classes[labels[i]]
        ann_label = classes[ann_pred[i]]
        snn_label = classes[snn_pred[i]]
        match = "✓" if ann_pred[i] == snn_pred[i] else "✗"
        print(f"   {match} True: {true_label:8s} | ANN: {ann_label:8s} | SNN: {snn_label:8s}")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Train full ANN:     python train_ann.py")
    print("  2. Convert to SNN:     python convert_to_snn.py")
    print("  3. Evaluate SNN:       python evaluate_snn.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        quick_demo()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")