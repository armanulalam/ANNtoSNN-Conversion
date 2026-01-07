"""
Utility Functions for ANN-SNN Conversion Project
Dataset loading, transformations, and helper functions.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_cifar10_loaders(batch_size=128, data_dir='./data', num_workers=2):
    """
    Load CIFAR-10 dataset with standard preprocessing.
    
    Args:
        batch_size: Batch size for training and testing
        data_dir: Directory to store/load CIFAR-10 data
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
        classes: Class names
    """
    import os
    import shutil
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    # Standard normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    # Try to load datasets with error handling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Load datasets
            train_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform_train
            )
            
            test_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform_test
            )
            break  # Success, exit retry loop
            
        except Exception as e:
            print(f"\n⚠ Download/load attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                print("Cleaning up corrupted data and retrying...")
                # Remove corrupted data
                cifar_path = os.path.join(data_dir, 'cifar-10-batches-py')
                if os.path.exists(cifar_path):
                    shutil.rmtree(cifar_path)
                    print(f"Removed corrupted data from {cifar_path}")
                
                # Remove tar file if it exists
                tar_file = os.path.join(data_dir, 'cifar-10-python.tar.gz')
                if os.path.exists(tar_file):
                    os.remove(tar_file)
                    print(f"Removed corrupted archive from {tar_file}")
            else:
                print("\n❌ Failed to download CIFAR-10 after multiple attempts.")
                print("\nTroubleshooting steps:")
                print("1. Check your internet connection")
                print("2. Manually delete the './data' folder and try again:")
                print("   rm -rf ./data")
                print("3. Download CIFAR-10 manually from:")
                print("   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
                print("4. Extract to './data/cifar-10-batches-py/'")
                raise
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"✓ CIFAR-10 loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    
    return train_loader, test_loader, classes


def evaluate_ann(model, test_loader, device):
    """
    Evaluate ANN model on test set.
    
    Args:
        model: ANN model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        accuracy: Test accuracy
        per_class_acc: Per-class accuracy dictionary
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    accuracy = 100 * correct / total
    per_class_acc = {i: 100 * class_correct[i] / class_total[i] 
                     for i in range(10) if class_total[i] > 0}
    
    return accuracy, per_class_acc


def evaluate_snn_timesteps(model, test_loader, device, timesteps_list):
    """
    Evaluate SNN at different timesteps to analyze accuracy-latency trade-off.
    
    Args:
        model: SNN model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        timesteps_list: List of timesteps to evaluate
        
    Returns:
        results: Dictionary with accuracy and spike counts for each timestep
    """
    model.eval()
    results = {}
    
    for num_steps in timesteps_list:
        model.num_steps = num_steps
        correct = 0
        total = 0
        total_spikes = 0
        spike_counts_all = {
            'layer1': 0, 'layer2': 0, 'layer3': 0, 
            'layer4': 0, 'layer5': 0
        }
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                spk_rec, mem_rec, spike_counts = model(images)
                
                # Prediction: sum spikes over time or use final membrane potential
                # We'll use sum of spikes for classification
                spike_sum = spk_rec.sum(dim=0)  # [batch_size, num_classes]
                _, predicted = torch.max(spike_sum, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Accumulate spike counts
                for key in spike_counts:
                    spike_counts_all[key] += spike_counts[key]
        
        accuracy = 100 * correct / total
        total_spikes = sum(spike_counts_all.values())
        avg_spikes_per_sample = total_spikes / len(test_loader.dataset)
        
        results[num_steps] = {
            'accuracy': accuracy,
            'total_spikes': total_spikes,
            'avg_spikes_per_sample': avg_spikes_per_sample,
            'spike_counts': spike_counts_all
        }
        
        print(f"  T={num_steps:3d} | Acc: {accuracy:.2f}% | "
              f"Avg Spikes/Sample: {avg_spikes_per_sample:.1f}")
    
    return results


def plot_accuracy_vs_timesteps(results, save_path='results/accuracy_vs_timesteps.png'):
    """
    Plot accuracy vs timesteps with spike count overlay.
    
    Args:
        results: Results dictionary from evaluate_snn_timesteps
        save_path: Path to save the plot
    """
    timesteps = sorted(results.keys())
    accuracies = [results[t]['accuracy'] for t in timesteps]
    avg_spikes = [results[t]['avg_spikes_per_sample'] for t in timesteps]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy
    color = 'tab:blue'
    ax1.set_xlabel('Number of Timesteps', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color=color, fontsize=12)
    ax1.plot(timesteps, accuracies, marker='o', linewidth=2, 
             markersize=8, color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot spike count on secondary axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg Spikes per Sample', color=color, fontsize=12)
    ax2.plot(timesteps, avg_spikes, marker='s', linewidth=2, 
             markersize=8, color=color, linestyle='--', label='Spike Count')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('SNN Performance: Accuracy vs Latency Trade-off', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    plt.close()


def plot_spike_distribution(results, timestep, save_path='results/spike_distribution.png'):
    """
    Plot spike distribution across layers for a specific timestep.
    
    Args:
        results: Results dictionary
        timestep: Specific timestep to visualize
        save_path: Path to save the plot
    """
    spike_counts = results[timestep]['spike_counts']
    
    layers = list(spike_counts.keys())
    counts = list(spike_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(layers, counts, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Total Spike Count', fontsize=12)
    plt.title(f'Spike Distribution Across Layers (T={timestep})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Spike distribution plot saved to {save_path}")
    plt.close()


def print_comparison_table(ann_acc, snn_results, timesteps=[10, 25, 50, 100]):
    """
    Print formatted comparison table of ANN vs SNN performance.
    
    Args:
        ann_acc: ANN accuracy
        snn_results: SNN results dictionary
        timesteps: List of timesteps to include in table
    """
    print("\n" + "="*70)
    print("Performance Comparison: ANN vs SNN")
    print("="*70)
    print(f"{'Model':<15} {'Timesteps':<12} {'Accuracy':<12} {'Spikes/Sample':<15}")
    print("-"*70)
    print(f"{'ANN (baseline)':<15} {'-':<12} {ann_acc:>10.2f}% {'-':<15}")
    print("-"*70)
    
    for t in timesteps:
        if t in snn_results:
            acc = snn_results[t]['accuracy']
            spikes = snn_results[t]['avg_spikes_per_sample']
            print(f"{'SNN':<15} {t:<12} {acc:>10.2f}% {spikes:>14.1f}")
    
    print("="*70)


def save_results_log(ann_acc, snn_results, save_path='results/logs.txt'):
    """
    Save detailed results to a log file.
    
    Args:
        ann_acc: ANN accuracy
        snn_results: SNN results dictionary
        save_path: Path to save log file
    """
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ANN to SNN Conversion - Experiment Results\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"ANN Baseline Accuracy: {ann_acc:.2f}%\n\n")
        
        f.write("SNN Performance at Different Timesteps:\n")
        f.write("-"*70 + "\n")
        
        for timestep in sorted(snn_results.keys()):
            result = snn_results[timestep]
            f.write(f"\nTimesteps: {timestep}\n")
            f.write(f"  Accuracy: {result['accuracy']:.2f}%\n")
            f.write(f"  Total Spikes: {result['total_spikes']:,}\n")
            f.write(f"  Avg Spikes/Sample: {result['avg_spikes_per_sample']:.2f}\n")
            f.write(f"  Spike Distribution:\n")
            for layer, count in result['spike_counts'].items():
                f.write(f"    {layer}: {count:,}\n")
    
    print(f"✓ Results log saved to {save_path}")


if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size=128)
    print(f"Classes: {classes}")
    print(f"Number of batches - Train: {len(train_loader)}, Test: {len(test_loader)}")