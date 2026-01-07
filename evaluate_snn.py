"""
Evaluate SNN Performance
Analyze accuracy-latency-energy trade-offs across different timesteps.
"""

import torch
import sys
from pathlib import Path
import numpy as np

# Add models to path
sys.path.append(str(Path(__file__).parent))

from models.ann_cnn import ANN_CNN
from models.snn_cnn import SNN_CNN
from utils import (
    get_cifar10_loaders, 
    evaluate_ann,
    evaluate_snn_timesteps,
    plot_accuracy_vs_timesteps,
    plot_spike_distribution,
    print_comparison_table,
    save_results_log
)


def comprehensive_snn_evaluation(snn_model, test_loader, device, 
                                 timesteps_list=[5, 10, 15, 20, 25, 30, 40, 50, 75, 100]):
    """
    Comprehensive evaluation of SNN at multiple timesteps.
    
    Args:
        snn_model: SNN model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        timesteps_list: List of timesteps to evaluate
        
    Returns:
        results: Dictionary of results at each timestep
    """
    
    print("\n" + "="*70)
    print("SNN Evaluation Across Multiple Timesteps")
    print("="*70)
    print(f"Device: {device}")
    print(f"Timesteps to evaluate: {timesteps_list}")
    print("="*70 + "\n")
    
    snn_model = snn_model.to(device)
    
    # Evaluate at each timestep
    print("Evaluating SNN performance...")
    print("-"*70)
    results = evaluate_snn_timesteps(snn_model, test_loader, device, timesteps_list)
    print("-"*70)
    
    return results


def analyze_energy_efficiency(results):
    """
    Analyze energy efficiency of SNN compared to theoretical ANN.
    
    Args:
        results: Results dictionary from SNN evaluation
    """
    
    print("\n" + "="*70)
    print("Energy Efficiency Analysis")
    print("="*70)
    
    # Theoretical energy comparison
    # Assumption: Each spike consumes ~1 unit of energy
    # Dense ANN computation: Every neuron activates every time (much higher energy)
    
    print("\nKey Insights:")
    print("-"*70)
    
    # Find optimal operating point (best accuracy with minimum timesteps)
    timesteps = sorted(results.keys())
    accuracies = [results[t]['accuracy'] for t in timesteps]
    spikes = [results[t]['avg_spikes_per_sample'] for t in timesteps]
    
    # Find first timestep that achieves >95% of max accuracy
    max_acc = max(accuracies)
    target_acc = 0.95 * max_acc
    
    optimal_t = None
    for t, acc in zip(timesteps, accuracies):
        if acc >= target_acc:
            optimal_t = t
            break
    
    if optimal_t:
        print(f"1. Optimal Operating Point:")
        print(f"   Timesteps: {optimal_t}")
        print(f"   Accuracy: {results[optimal_t]['accuracy']:.2f}%")
        print(f"   Spikes/Sample: {results[optimal_t]['avg_spikes_per_sample']:.1f}")
        print(f"   (Achieves {target_acc:.1f}% of maximum accuracy)")
    
    # Spike sparsity analysis
    print(f"\n2. Spike Sparsity:")
    min_spikes = min(spikes)
    max_spikes = max(spikes)
    print(f"   Min spikes/sample: {min_spikes:.1f} (T={timesteps[spikes.index(min_spikes)]})")
    print(f"   Max spikes/sample: {max_spikes:.1f} (T={timesteps[spikes.index(max_spikes)]})")
    print(f"   Sparsity ratio: {min_spikes/max_spikes:.2f}")
    
    # Energy-accuracy trade-off
    print(f"\n3. Energy-Accuracy Trade-off:")
    for t in [10, 25, 50, 100]:
        if t in results:
            acc = results[t]['accuracy']
            spk = results[t]['avg_spikes_per_sample']
            efficiency = acc / spk  # Accuracy per spike (higher is better)
            print(f"   T={t:3d}: {acc:6.2f}% accuracy, {spk:8.1f} spikes, "
                  f"Efficiency: {efficiency:.4f}")
    
    print("="*70 + "\n")


def analyze_convergence(results):
    """
    Analyze how SNN accuracy converges over time.
    
    Args:
        results: Results dictionary from SNN evaluation
    """
    
    print("\n" + "="*70)
    print("Convergence Analysis")
    print("="*70)
    
    timesteps = sorted(results.keys())
    accuracies = [results[t]['accuracy'] for t in timesteps]
    
    # Calculate accuracy improvements
    print("\nAccuracy Improvement per Timestep:")
    print("-"*70)
    for i, t in enumerate(timesteps):
        if i == 0:
            improvement = accuracies[i]
            print(f"  T={t:3d}: {accuracies[i]:6.2f}% (baseline)")
        else:
            improvement = accuracies[i] - accuracies[i-1]
            print(f"  T={t:3d}: {accuracies[i]:6.2f}% (+{improvement:5.2f}%)")
    
    # Check for diminishing returns
    print("\nDiminishing Returns Analysis:")
    print("-"*70)
    if len(timesteps) >= 3:
        early_gain = accuracies[1] - accuracies[0]
        late_gain = accuracies[-1] - accuracies[-2]
        print(f"  Early gain (T={timesteps[0]}→{timesteps[1]}): +{early_gain:.2f}%")
        print(f"  Late gain (T={timesteps[-2]}→{timesteps[-1]}): +{late_gain:.2f}%")
        
        if late_gain < 0.5:
            print("  ✓ Convergence achieved - minimal gains from more timesteps")
        else:
            print("  → More timesteps may still improve accuracy")
    
    print("="*70 + "\n")


def main():
    """Main evaluation function."""
    
    # Configuration
    ANN_PATH = 'ann_cnn.pth'
    SNN_PATH = 'snn_cnn.pth'
    BATCH_SIZE = 128
    TIMESTEPS_LIST = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if models exist
    if not Path(ANN_PATH).exists():
        print(f"\n❌ Error: ANN model not found at {ANN_PATH}")
        print("Please train the ANN first: python train_ann.py")
        return
    
    if not Path(SNN_PATH).exists():
        print(f"\n❌ Error: SNN model not found at {SNN_PATH}")
        print("Please convert ANN to SNN first: python convert_to_snn.py")
        return
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    _, test_loader, classes = get_cifar10_loaders(batch_size=BATCH_SIZE)
    
    # Load ANN model for comparison
    print("\nLoading ANN baseline...")
    ann_model = ANN_CNN(num_classes=10)
    ann_model.load_state_dict(torch.load(ANN_PATH))
    ann_acc, _ = evaluate_ann(ann_model.to(device), test_loader, device)
    print(f"ANN Baseline Accuracy: {ann_acc:.2f}%")
    
    # Load SNN model
    print("\nLoading SNN model...")
    snn_model = SNN_CNN(num_classes=10, beta=0.95, num_steps=25)
    snn_model.load_state_dict(torch.load(SNN_PATH))
    print("✓ SNN model loaded")
    
    # Comprehensive SNN evaluation
    snn_results = comprehensive_snn_evaluation(
        snn_model=snn_model,
        test_loader=test_loader,
        device=device,
        timesteps_list=TIMESTEPS_LIST
    )
    
    # Analysis
    analyze_energy_efficiency(snn_results)
    analyze_convergence(snn_results)
    
    # Print comparison table
    print_comparison_table(ann_acc, snn_results, timesteps=[10, 25, 50, 100])
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_vs_timesteps(snn_results, save_path='results/accuracy_vs_timesteps.png')
    plot_spike_distribution(snn_results, timestep=25, save_path='results/spike_distribution.png')
    
    # Save results log
    save_results_log(ann_acc, snn_results, 'results/logs.txt')
    
    # Final summary
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print("Generated files:")
    print("  • results/accuracy_vs_timesteps.png")
    print("  • results/spike_distribution.png")
    print("  • results/logs.txt")
    print("\nKey Findings:")
    
    # Best SNN accuracy
    best_t = max(snn_results.keys(), key=lambda t: snn_results[t]['accuracy'])
    best_acc = snn_results[best_t]['accuracy']
    acc_gap = ann_acc - best_acc
    
    print(f"  • ANN Accuracy: {ann_acc:.2f}%")
    print(f"  • Best SNN Accuracy: {best_acc:.2f}% at T={best_t}")
    print(f"  • Accuracy Gap: {acc_gap:.2f}%")
    
    # Energy efficiency
    spikes_at_best = snn_results[best_t]['avg_spikes_per_sample']
    print(f"  • Avg Spikes/Sample (T={best_t}): {spikes_at_best:.1f}")
    print(f"  • Sparse spike-based computation demonstrates energy efficiency")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()