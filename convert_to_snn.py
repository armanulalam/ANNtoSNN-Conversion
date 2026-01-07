"""
Convert ANN to SNN
Transfer trained ANN weights to SNN architecture.
"""

import torch
import sys
from pathlib import Path

# Add models to path
sys.path.append(str(Path(__file__).parent))

from models.ann_cnn import ANN_CNN
from models.snn_cnn import SNN_CNN
from utils import get_cifar10_loaders, evaluate_ann


def convert_ann_to_snn(ann_path='ann_cnn.pth', snn_path='snn_cnn.pth',
                       beta=0.95, num_steps=25):
    """
    Convert trained ANN model to SNN by transferring weights.
    
    Args:
        ann_path: Path to trained ANN model
        snn_path: Path to save converted SNN model
        beta: Decay rate for LIF neurons
        num_steps: Default number of timesteps for SNN
        
    Returns:
        ann_model: Loaded ANN model
        snn_model: Converted SNN model
    """
    
    print("\n" + "="*70)
    print("ANN → SNN Conversion")
    print("="*70)
    
    # Check if ANN model exists
    if not Path(ann_path).exists():
        raise FileNotFoundError(
            f"ANN model not found at {ann_path}. "
            "Please train the ANN first using train_ann.py"
        )
    
    # Load ANN model
    print(f"\n1. Loading trained ANN from {ann_path}...")
    ann_model = ANN_CNN(num_classes=10)
    ann_model.load_state_dict(torch.load(ann_path))
    ann_model.eval()
    print("   ✓ ANN model loaded successfully")
    
    # Create SNN model
    print(f"\n2. Creating SNN architecture...")
    print(f"   Beta (membrane decay): {beta}")
    print(f"   Default timesteps: {num_steps}")
    snn_model = SNN_CNN(num_classes=10, beta=beta, num_steps=num_steps)
    print("   ✓ SNN model created")
    
    # Transfer weights
    print(f"\n3. Transferring weights from ANN to SNN...")
    snn_model.load_ann_weights(ann_model)
    
    # Save converted SNN
    print(f"\n4. Saving converted SNN to {snn_path}...")
    torch.save(snn_model.state_dict(), snn_path)
    print(f"   ✓ SNN model saved")
    
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"ANN model: {ann_path}")
    print(f"SNN model: {snn_path}")
    print("\nNext steps:")
    print("  1. Run evaluate_snn.py to analyze SNN performance")
    print("  2. Compare accuracy vs latency trade-offs")
    print("  3. Analyze spike activity and energy efficiency")
    print("="*70 + "\n")
    
    return ann_model, snn_model


def verify_conversion(ann_model, snn_model, device='cpu', num_samples=100):
    """
    Verify that weight transfer was successful by comparing layer outputs.
    
    Args:
        ann_model: ANN model
        snn_model: SNN model
        device: Device to run verification on
        num_samples: Number of samples to test
    """
    print("\n" + "="*70)
    print("Verification: Comparing ANN and SNN Layer Outputs")
    print("="*70)
    
    ann_model = ann_model.to(device)
    snn_model = snn_model.to(device)
    ann_model.eval()
    snn_model.eval()
    
    # Load a small batch of data
    _, test_loader, _ = get_cifar10_loaders(batch_size=num_samples)
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        # ANN forward pass
        ann_output = ann_model(images)
        ann_pred = torch.argmax(ann_output, dim=1)
        
        # SNN forward pass (multiple timesteps)
        spk_rec, mem_rec, _ = snn_model(images)
        snn_output = spk_rec.sum(dim=0)  # Sum spikes over time
        snn_pred = torch.argmax(snn_output, dim=1)
        
        # Compare predictions
        agreement = (ann_pred == snn_pred).float().mean().item() * 100
        
        print(f"\nPrediction Agreement: {agreement:.2f}%")
        print(f"(Percentage of samples where ANN and SNN agree)")
        
        if agreement > 80:
            print("✓ Good agreement - conversion likely successful")
        elif agreement > 60:
            print("⚠ Moderate agreement - some timestep tuning may be needed")
        else:
            print("✗ Low agreement - check conversion process")
    
    print("="*70 + "\n")


def main():
    """Main conversion function."""
    
    # Configuration
    ANN_PATH = 'ann_cnn.pth'
    SNN_PATH = 'snn_cnn.pth'
    BETA = 0.95  # Membrane potential decay rate
    NUM_STEPS = 25  # Default inference timesteps
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert ANN to SNN
    ann_model, snn_model = convert_ann_to_snn(
        ann_path=ANN_PATH,
        snn_path=SNN_PATH,
        beta=BETA,
        num_steps=NUM_STEPS
    )
    
    # Verify conversion
    print("\nVerifying conversion...")
    verify_conversion(ann_model, snn_model, device=device, num_samples=100)
    
    # Quick evaluation on test set
    print("\nQuick evaluation on CIFAR-10 test set...")
    _, test_loader, _ = get_cifar10_loaders(batch_size=128)
    ann_acc, _ = evaluate_ann(ann_model.to(device), test_loader, device)
    print(f"\nANN Test Accuracy: {ann_acc:.2f}%")
    print("\nFor detailed SNN evaluation at multiple timesteps, run:")
    print("  python evaluate_snn.py")


if __name__ == "__main__":
    main()