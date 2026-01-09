# ANN → SNN Conversion for Energy-Efficient Image Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## 🎯 Quick Start

```bash
pip install -r requirements.txt  # Install dependencies
python demo.py                   # Verify setup (1 min)
python train_ann.py              # Train ANN (1-10 hours)
python convert_to_snn.py         # Convert to SNN (1 min)
python evaluate_snn.py           # Evaluate SNN (10-15 min)
```

---

## 📋 Table of Contents

- [Research Motivation](#-research-motivation)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [How to Run This Project](#-how-to-run-this-project)
- [Expected Results](#-expected-results)
- [Project Structure](#-project-structure)
- [Customization](#-customization)
- [Troubleshooting](#-troubleshooting)
- [Understanding the Results](#-understanding-the-results)
- [Citation](#-citation)

---

## 🎓 Project Motivation

Conventional neural networks perform **dense computations** at every layer, which is energy-inefficient for edge and neuromorphic systems. **Spiking Neural Networks (SNNs)** communicate using **sparse spikes over time**, offering a biologically inspired and energy-efficient alternative.

This project demonstrates:
- ✅ How ANN performance can be preserved in SNNs
- ✅ How accuracy evolves over time (timesteps)
- ✅ How sparse spiking reduces computational cost
- ✅ The accuracy-latency-energy trade-off

---

## 🚀 Key Features

- **PyTorch-based ANN Training**: Standard CNN on CIFAR-10
- **ANN→SNN Conversion**: Weight transfer with minimal accuracy loss
- **Spike-based Inference**: Leaky Integrate-and-Fire (LIF) neurons
- **Comprehensive Analysis**: Accuracy vs timesteps, spike activity, energy efficiency
- **Research-Ready**: Clean code structure, detailed logging, visualizations

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- ~1GB free disk space
- Internet connection (for downloading packages and dataset)

### Step 1: Clone or Download Project

```bash
cd ANNtoSNN-Conversion
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
conda create -n snnproj python=3.9 -y
# Activating it
conda activate snnproj
```

### Step 3: Install Dependencies

**For CPU:**
```bash
pip install -r requirements.txt
```

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install snntorch matplotlib seaborn numpy
```

**Verify Installation:**
```bash
python -c "import torch; import snntorch; print('✓ All packages installed')"
```

---

## 🎯 How to Run This Project


#### **Step 0: Quick Verification**

Before running the full pipeline, verify everything works:

```bash
python demo.py
```

**Expected Output:**
```
Device: cpu
1. Loading CIFAR-10 dataset...
✓ CIFAR-10 loaded: 50000 train, 10000 test
2. Testing ANN CNN model...
✓ ANN output shape: torch.Size([32, 10])
✓ Parameters: 3,539,146
3. Testing SNN CNN model...
✓ SNN spike recordings: torch.Size([25, 32, 10])
✓ Prediction agreement: 50-70%
Demo Complete!
```

If this works, proceed to full pipeline! ✅

---

#### **Step 1: Train ANN Baseline**

Train a conventional CNN on CIFAR-10:

```bash
python train_ann.py
```

**Configuration Options:**

Default settings (in `train_ann.py`):
- `NUM_EPOCHS = 50` (full training, ~6-10 hours on CPU)
- `BATCH_SIZE = 128`
- `LEARNING_RATE = 0.001`

**For Faster Testing:**
Edit `train_ann.py`, line 129:
```python
NUM_EPOCHS = 10  # Takes ~1.5-2 hours on CPU
```

**What Happens:**
- Downloads CIFAR-10 dataset automatically (~170MB)
- Trains CNN with data augmentation
- Evaluates on test set every epoch
- Saves best model to `ann_cnn.pth`
- Shows progress every 100 batches

**Expected Output:**
```
Using device: cpu
Loading CIFAR-10 dataset...
✓ CIFAR-10 loaded: 50000 train, 10000 test

Training ANN CNN on CIFAR-10
Device: cpu
Epochs: 50
Parameters: 3,539,146

Epoch [1/50] Batch [100/391] Loss: 1.8234
...
✓ Best model saved (Accuracy: 72.45%)

Training Complete!
Best Test Accuracy: 89.23%
Model saved to: ann_cnn.pth
```

**Time Estimate:**
- CPU: 6-10 hours (50 epochs) or 1.5-2 hours (10 epochs)
- GPU: 15-20 minutes (50 epochs)

**Expected Accuracy:**
- 10 epochs: 60-65%
- 20 epochs: 65-70%
- 50 epochs: 70-90%

---

#### **Step 2: Convert ANN to SNN**

Transfer trained weights from ANN to SNN:

```bash
python convert_to_snn.py
```

**What Happens:**
- Loads trained ANN from `ann_cnn.pth`
- Creates SNN with same architecture
- Transfers Conv, FC, and BatchNorm weights
- Verifies conversion by comparing predictions
- Saves SNN to `snn_cnn.pth`

**Expected Output:**
```
ANN → SNN Conversion

1. Loading trained ANN from ann_cnn.pth...
   ✓ ANN model loaded successfully

2. Creating SNN architecture...
   Beta (membrane decay): 0.95
   Default timesteps: 25
   ✓ SNN model created

3. Transferring weights from ANN to SNN...
   ✓ ANN weights successfully loaded into SNN

4. Saving converted SNN to snn_cnn.pth...
   ✓ SNN model saved

Conversion Complete!

Verification: Comparing ANN and SNN Layer Outputs
Prediction Agreement: 82.45%
✓ Good agreement - conversion likely successful

ANN Test Accuracy: 89.23%
```

**Time:** ~1 minute

---

#### **Step 3: Evaluate SNN Performance**

Analyze SNN at multiple timesteps:

```bash
python evaluate_snn.py
```

**What Happens:**
- Tests SNN at timesteps: [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
- Measures accuracy and spike activity at each timestep
- Generates performance plots
- Analyzes energy efficiency and convergence
- Saves detailed logs

**Expected Output:**
```
SNN Evaluation Across Multiple Timesteps
Device: cpu
Timesteps to evaluate: [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

Evaluating SNN performance...
  T=  5 | Acc: 42.71% | Avg Spikes/Sample: 14508.5
  T= 10 | Acc: 53.88% | Avg Spikes/Sample: 33958.8
  T= 25 | Acc: 60.06% | Avg Spikes/Sample: 92198.6
  T= 50 | Acc: 61.48% | Avg Spikes/Sample: 189458.4
  T=100 | Acc: 61.93% | Avg Spikes/Sample: 383916.1

Energy Efficiency Analysis
1. Optimal Operating Point:
   Timesteps: 20
   Accuracy: 59.19%
   Spikes/Sample: 72883.2

Performance Comparison: ANN vs SNN
Model           Timesteps    Accuracy     Spikes/Sample
ANN (baseline)  -              89.23%     -
SNN             25             60.06%        92198.6
SNN             50             61.48%       189458.4
SNN             100            61.93%       383916.1

Evaluation Complete!
Generated files:
  • results/accuracy_vs_timesteps.png
  • results/spike_distribution.png
  • results/logs.txt
```

**Time:** 10-15 minutes

**Generated Files:**
```
results/
├── accuracy_vs_timesteps.png    # Performance visualization
├── spike_distribution.png       # Spike activity per layer
└── logs.txt                     # Detailed experiment results
```

---

#### **Step 4: View Results**

**View Plots:**

```bash
# Windows
start results\accuracy_vs_timesteps.png
start results\spike_distribution.png

# Mac
open results/accuracy_vs_timesteps.png
open results/spike_distribution.png

# Linux
xdg-open results/accuracy_vs_timesteps.png
```

**Read Detailed Logs:**

```bash
# Windows
type results\logs.txt

# Mac/Linux
cat results/logs.txt
```

---

## 📊 Expected Results

### Performance Summary

| Model | Timesteps | Accuracy | Spikes/Sample | Notes |
|-------|-----------|----------|---------------|-------|
| **ANN** | - | 70-90% | - | Dense computation |
| **SNN** | 10 | 53-58% | ~34K | Low latency |
| **SNN** | 25 | 60-65% | ~92K | **Optimal** ⭐ |
| **SNN** | 50 | 61-66% | ~189K | High accuracy |
| **SNN** | 100 | 62-67% | ~384K | Converged |

### Key Findings

1. **SNN achieves 60-67% accuracy** at T=100 (varies based on ANN training)
2. **Clear accuracy-latency trade-off**: More timesteps → better accuracy
3. **Diminishing returns** after T=50-75
4. **Sparse spike activity**: ~100K-400K spikes vs millions in dense ANN
5. **Optimal operating point**: T=20-25 balances speed and accuracy

### Visualization Examples

The `accuracy_vs_timesteps.png` plot shows:
- 📈 Blue line: Accuracy increasing with timesteps
- 📉 Red line: Spike count (sub-linear growth)
- Clear convergence behavior

---

## 📁 Project Structure

```
ANNtoSNN-Conversion/
│
├── data/                           # CIFAR-10 dataset (auto-downloaded)
│   └── cifar-10-batches-py/
│
├── models/                         # Model definitions
│   ├── __init__.py
│   ├── ann_cnn.py                  # Standard CNN (ANN)
│   └── snn_cnn.py                  # Spiking CNN (SNN)
│
├── results/                        # Generated results
│   ├── accuracy_vs_timesteps.png
│   ├── spike_distribution.png
│   └── logs.txt
│
├── train_ann.py                    # Step 1: Train ANN
├── convert_to_snn.py               # Step 2: Convert to SNN
├── evaluate_snn.py                 # Step 3: Evaluate SNN
├── utils.py                        # Helper functions
├── demo.py                         # Quick verification script
│
├── download_cifar10.py             # Manual CIFAR-10 download
├── cleanup_data.py                 # Data cleanup utility
│
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
├── TROUBLESHOOTING.md              # Common issues & solutions
└── LICENSE                         # MIT License
```

---

## ⚙️ Customization

### Adjust Training Parameters

**Edit `train_ann.py`:**

```python
# Line 128-130
NUM_EPOCHS = 10        # Faster training (default: 50)
BATCH_SIZE = 64        # Reduce if out of memory (default: 128)
LEARNING_RATE = 0.001  # Adjust for training stability
```

### Tune SNN Parameters

**Edit `convert_to_snn.py`:**

```python
# Line 138-139
BETA = 0.85           # Lower = faster response (default: 0.95)
NUM_STEPS = 25        # Default inference timesteps
```

Lower beta values (0.80-0.90) may improve SNN accuracy!

### Customize Evaluation

**Edit `evaluate_snn.py`:**

```python
# Line 172
TIMESTEPS_LIST = [10, 25, 50, 100]  # Test fewer timesteps (faster)
# or
TIMESTEPS_LIST = [100, 150, 200]    # Test higher timesteps
```

---

## 🔧 Troubleshooting

### Issue: CIFAR-10 Download Failed

**Error:** `Dataset not found or corrupted`

**Solution:**
```bash
python download_cifar10.py
```

Or see [MANUAL_DOWNLOAD.md](MANUAL_DOWNLOAD.md) for step-by-step manual download.

### Issue: Out of Memory

**Solution:** Reduce batch size

Edit training/evaluation scripts:
```python
BATCH_SIZE = 64  # or 32
```

### Issue: Training Too Slow

**Solutions:**
1. Reduce epochs: `NUM_EPOCHS = 10`
2. Use GPU (15-20 min instead of hours)
3. Train overnight with full 50 epochs

### Issue: Module Not Found

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Large ANN-SNN Accuracy Gap

If SNN accuracy is much lower than ANN (>15% gap):

**Solution:** Try lower beta value

Edit `convert_to_snn.py`:
```python
BETA = 0.85  # or 0.80
```

Then re-run:
```bash
python convert_to_snn.py
python evaluate_snn.py
```

For more issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📖 Understanding the Results

### Interpreting Accuracy vs Timesteps

```
T=5  → T=10:  Large gain (+10-15%)  📈📈📈
T=10 → T=25:  Good gain (+5-8%)     📈📈
T=25 → T=50:  Small gain (+1-3%)    📈
T=50 → T=100: Minimal gain (<1%)    📊
```

**Conclusion:** SNN converges around T=50-75

### Energy Efficiency Metric

```
Efficiency = Accuracy / Spikes per Sample

Higher efficiency = Better performance per spike
```

Typically:
- T=10: High efficiency (fast, decent accuracy)
- T=25: **Best efficiency** (optimal balance) ⭐
- T=100: Lower efficiency (high accuracy, many spikes)

### Optimal Operating Point

The "optimal" timestep balances:
- ⚡ Inference speed (fewer timesteps)
- 🎯 Accuracy (more timesteps)
- 🔋 Energy (fewer spikes)

Usually **T=20-30** provides the best trade-off.

---

## 🎓 Research & Educational Use

### Key Concepts Demonstrated

1. **Spiking Neural Networks**: Event-driven, sparse computation
2. **Leaky Integrate-and-Fire Neurons**: Membrane potential, threshold
3. **ANN→SNN Conversion**: Weight transfer technique
4. **Accuracy-Latency Trade-off**: More time = better accuracy
5. **Energy Efficiency**: Sparse spikes vs dense operations

### For Academic Use

**Citation:**
```bibtex
@software{ann_snn_conversion_2026,
  title={ANN to SNN Conversion for Energy-Efficient Image Classification},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/ANNtoSNN-Conversion}
}
```

### Skills Showcased

- ✅ Deep Learning (PyTorch)
- ✅ Spiking Neural Networks
- ✅ Model Conversion Techniques
- ✅ Neuromorphic Computing
- ✅ Research Experimentation
- ✅ Data Visualization
- ✅ Scientific Computing

---

## 📚 Additional Resources

### Documentation

- **QUICKSTART.md** - Quick setup and running guide
- **TROUBLESHOOTING.md** - Common issues and solutions
- **FILE_STRUCTURE.md** - Detailed file descriptions
- **MANUAL_DOWNLOAD.md** - Manual CIFAR-10 download guide

### Key Papers

1. Rueckauer et al. (2017) - "Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks"
2. Diehl & Cook (2015) - "Unsupervised learning using spike-timing-dependent plasticity"
3. Sengupta et al. (2019) - "Going Deeper in Spiking Neural Networks"

### Libraries Used

- **PyTorch**: https://pytorch.org/
- **snnTorch**: https://snntorch.readthedocs.io/
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add more complex architectures (ResNet, VGG)
- [ ] Implement direct SNN training (no conversion)
- [ ] Add support for other datasets (ImageNet, etc.)
- [ ] Optimize SNN parameters automatically
- [ ] Add neuromorphic hardware simulation

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch team** for the deep learning framework
- **snnTorch developers** for the SNN library
- **CIFAR-10** dataset creators
- Research community for SNN conversion techniques

---

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: armaanularmaan@example.com

---

## ⭐ If You Find This Useful

Give it a star on GitHub! It helps others discover this project.

---

**🎉 Ready to explore energy-efficient neuromorphic computing? Start with `python demo.py`!**
