"""
SNN CNN Model for CIFAR-10 Classification
Spiking Neural Network with Leaky Integrate-and-Fire neurons.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN_CNN(nn.Module):
    """
    Spiking Convolutional Neural Network for CIFAR-10 classification.
    
    Architecture mirrors ANN_CNN but uses LIF neurons instead of ReLU.
    Conversion-friendly design for ANN→SNN weight transfer.
    """
    
    def __init__(self, num_classes=10, beta=0.95, num_steps=25):
        super(SNN_CNN, self).__init__()
        
        self.num_steps = num_steps
        self.beta = beta  # Decay rate for LIF neurons
        
        # Surrogate gradient for backpropagation through spikes
        spike_grad = surrogate.fast_sigmoid()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(512, num_classes)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        """
        Forward pass with spike-based computation.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            spk_rec: Spike recordings over time [num_steps, batch_size, num_classes]
            mem_rec: Membrane potential recordings [num_steps, batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Initialize hidden states for LIF neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        
        # Record spikes and membrane potentials over time
        spk_rec = []
        mem_rec = []
        
        # Spike counting for each layer (for analysis)
        spike_counts = {
            'layer1': 0, 'layer2': 0, 'layer3': 0, 
            'layer4': 0, 'layer5': 0
        }
        
        # Time-step loop
        for step in range(self.num_steps):
            # Convolutional Block 1
            cur1 = self.conv1(x)
            cur1 = self.bn1(cur1)
            cur1 = self.pool(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            spike_counts['layer1'] += spk1.sum().item()
            
            # Convolutional Block 2
            cur2 = self.conv2(spk1)
            cur2 = self.bn2(cur2)
            cur2 = self.pool(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spike_counts['layer2'] += spk2.sum().item()
            
            # Convolutional Block 3
            cur3 = self.conv3(spk2)
            cur3 = self.bn3(cur3)
            cur3 = self.pool(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            spike_counts['layer3'] += spk3.sum().item()
            
            # Flatten
            spk3_flat = spk3.view(batch_size, -1)
            
            # Fully connected layer 1
            cur4 = self.fc1(spk3_flat)
            cur4 = self.bn_fc1(cur4)
            spk4, mem4 = self.lif4(cur4, mem4)
            spike_counts['layer4'] += spk4.sum().item()
            
            # Fully connected layer 2 (output)
            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            spike_counts['layer5'] += spk5.sum().item()
            
            spk_rec.append(spk5)
            mem_rec.append(mem5)
        
        # Stack recordings
        spk_rec = torch.stack(spk_rec, dim=0)
        mem_rec = torch.stack(mem_rec, dim=0)
        
        return spk_rec, mem_rec, spike_counts
    
    def load_ann_weights(self, ann_model):
        """
        Load weights from trained ANN model.
        
        Args:
            ann_model: Trained ANN_CNN model
        """
        # Load convolutional and linear layer weights
        self.conv1.weight.data = ann_model.conv1.weight.data.clone()
        self.conv1.bias.data = ann_model.conv1.bias.data.clone()
        self.bn1.weight.data = ann_model.bn1.weight.data.clone()
        self.bn1.bias.data = ann_model.bn1.bias.data.clone()
        self.bn1.running_mean.data = ann_model.bn1.running_mean.data.clone()
        self.bn1.running_var.data = ann_model.bn1.running_var.data.clone()
        
        self.conv2.weight.data = ann_model.conv2.weight.data.clone()
        self.conv2.bias.data = ann_model.conv2.bias.data.clone()
        self.bn2.weight.data = ann_model.bn2.weight.data.clone()
        self.bn2.bias.data = ann_model.bn2.bias.data.clone()
        self.bn2.running_mean.data = ann_model.bn2.running_mean.data.clone()
        self.bn2.running_var.data = ann_model.bn2.running_var.data.clone()
        
        self.conv3.weight.data = ann_model.conv3.weight.data.clone()
        self.conv3.bias.data = ann_model.conv3.bias.data.clone()
        self.bn3.weight.data = ann_model.bn3.weight.data.clone()
        self.bn3.bias.data = ann_model.bn3.bias.data.clone()
        self.bn3.running_mean.data = ann_model.bn3.running_mean.data.clone()
        self.bn3.running_var.data = ann_model.bn3.running_var.data.clone()
        
        self.fc1.weight.data = ann_model.fc1.weight.data.clone()
        self.fc1.bias.data = ann_model.fc1.bias.data.clone()
        self.bn_fc1.weight.data = ann_model.bn_fc1.weight.data.clone()
        self.bn_fc1.bias.data = ann_model.bn_fc1.bias.data.clone()
        self.bn_fc1.running_mean.data = ann_model.bn_fc1.running_mean.data.clone()
        self.bn_fc1.running_var.data = ann_model.bn_fc1.running_var.data.clone()
        
        self.fc2.weight.data = ann_model.fc2.weight.data.clone()
        self.fc2.bias.data = ann_model.fc2.bias.data.clone()
        
        print("✓ ANN weights successfully loaded into SNN")


if __name__ == "__main__":
    # Test the model
    model = SNN_CNN(num_classes=10, num_steps=25)
    x = torch.randn(2, 3, 32, 32)  # Batch of 2 CIFAR-10 images
    spk_rec, mem_rec, spike_counts = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Spike recordings shape: {spk_rec.shape}")
    print(f"Membrane recordings shape: {mem_rec.shape}")
    print(f"Spike counts per layer: {spike_counts}")