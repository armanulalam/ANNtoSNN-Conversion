"""
Train ANN CNN on CIFAR-10
Establishes the baseline performance for comparison with SNN.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import sys
from pathlib import Path

# Add models to path
sys.path.append(str(Path(__file__).parent))

from models.ann_cnn import ANN_CNN
from utils import get_cifar10_loaders, evaluate_ann


def train_ann(model, train_loader, test_loader, device, 
              num_epochs=50, learning_rate=0.001):
    """
    Train ANN CNN model on CIFAR-10.
    
    Args:
        model: ANN model to train
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
    """
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0.0
    train_losses = []
    test_accuracies = []
    
    print("\n" + "="*70)
    print("Training ANN CNN on CIFAR-10")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Parameters: {model.get_num_parameters():,}")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{i+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        # Epoch statistics
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        test_acc, per_class_acc = evaluate_ann(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'ann_cnn.pth')
            print(f"  ✓ Best model saved (Accuracy: {best_acc:.2f}%)")
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print("-"*70 + "\n")
    
    # Training complete
    total_time = time.time() - start_time
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: ann_cnn.pth")
    print("="*70 + "\n")
    
    return train_losses, test_accuracies, best_acc


def main():
    """Main training function."""
    
    # Configuration
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_WORKERS = 2
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader, classes = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Create model
    print("\nInitializing ANN CNN model...")
    model = ANN_CNN(num_classes=10)
    
    # Train model
    train_losses, test_accuracies, best_acc = train_ann(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    model.load_state_dict(torch.load('ann_cnn.pth'))
    final_acc, per_class_acc = evaluate_ann(model, test_loader, device)
    
    print("\nPer-Class Accuracy:")
    print("-"*40)
    for i, class_name in enumerate(classes):
        print(f"  {class_name:10s}: {per_class_acc[i]:.2f}%")
    print("-"*40)
    print(f"  Average: {final_acc:.2f}%")
    print()


if __name__ == "__main__":
    main()