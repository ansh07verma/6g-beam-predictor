# 6G Beam Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance, end-to-end machine learning pipeline for predicting optimal beamforming directions in 6G massive MIMO systems using solely User Equipment (UE) coordinates. This project leverages real-world mmWave ray-tracing data from the **DeepMIMO** database to frame beam selection as a classification problem over a DFT codebook, eliminating the need for exhaustive beam sweeping.

## 🚀 Key Features

- **High Accuracy**: Achieves **>87% Top-1 accuracy** and **100% Top-5 accuracy** on the 28 GHz mmWave dataset.
- **Near-Optimal Performance**: Retains **>98% of optimal achievable channel capacity** (Rate Ratio: 0.9866 at 20 dB SNR).
- **Efficient Architecture**: Deep Residual MLP (~1.2M parameters) with pre-activated skip connections and GELU activations for stable gradient flow.
- **Automated Pipeline**: Handles dataset downloading, preprocessing, training, evaluation, and visualization out-of-the-box.
- **Security Hardened**: Implements `weights_only=True` for safe checkpoint loading, input validation, and path sanitization.

## 📊 Performance Metrics (o1_28 Scenario)

| Metric | Result | Target | Status |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **87.94%** | > 85% | ✅ PASS |
| **Top-5 Accuracy** | **100.00%** | > 97% | ✅ PASS |
| **Rate Ratio** | **0.9866** | > 0.95 | ✅ PASS |

*Evaluated at SNR = 20 dB. Training completed in ~6 minutes on CPU.*

## 🏗️ Model Architecture

The `BeamPredictor` is a Deep Residual MLP designed for spatial regression and classification:

1. **Input**: 6D normalized UE features $(x, y, z, r, \text{azimuth}, \text{elevation})$
2. **Stem**: `Linear(512)` → `BatchNorm` → `GELU`
3. **Trunk**: 2 × Pre-activated Residual Blocks (dim=512)
4. **Neck**: `Linear(256)` → `BN` → `GELU` → `Dropout(0.2)` → `Linear(128)` → `BN` → `GELU` → `Dropout(0.1)`
5. **Head**: `Linear(64)` mapping to unnormalized beam logits

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/6g-beam-predictor.git
cd 6g-beam-predictor
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install PyTorch (CPU or GPU)
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install -r requirements.txt
```

## 📦 Dataset

This project uses the **`o1_28`** (Outdoor-1, 28 GHz mmWave) scenario from the DeepMIMO v4 database. 
- **Environment**: Outdoor urban campus
- **Positions**: ~500,000 UE locations (filtered to ~21,500 valid line-of-sight/non-blocked users)
- **Antennas**: 64-element Uniform Linear Array (ULA) at Base Station, 1 antenna at UE
- **Classes**: 64 predefined DFT beam directions

*The dataset downloads automatically when you first run the training or evaluation script. If you encounter rate limits, use the provided `notebooks/DeepMIMO_O1_28_Downloader.ipynb` in Google Colab and place the extracted folder in `deepmimo_scenarios/`.*

## ⚡ Quick Start

### 1. Train the Model
Train the Deep Residual MLP on the real dataset (defaults to 200 epochs, batch size 256):
```bash
python src/train.py --scenario o1_28 --epochs 200 --batch-size 256 --device cpu
```
*Checkpoints and training history are saved to `checkpoints/`.*

### 2. Evaluate the Model
Evaluate the best checkpoint against PRD metrics (Top-1 Acc, Top-5 Acc, Rate Ratio) and generate visualization plots:
```bash
python src/evaluate.py --real --scenario o1_28 --checkpoint checkpoints/best_mlp.pt
```
*Plots are saved to `results/`.*

### 3. Explore the Data
Open the Jupyter notebooks for interactive data exploration and result visualization:
```bash
jupyter notebook notebooks/01_explore_data.ipynb
jupyter notebook notebooks/02_results.ipynb
```

## 📁 Project Structure

```text
6g-beam-predictor/
├── checkpoints/          # Saved model weights (.pt) and training history (.json)
├── data/                 # (Optional) Local dataset storage
├── deepmimo_scenarios/   # Auto-downloaded DeepMIMO scenario files
├── notebooks/            # Jupyter notebooks for exploration and visualization
│   ├── 01_explore_data.ipynb
│   ├── 02_results.ipynb
│   └── DeepMIMO_O1_28_Downloader.ipynb
├── results/              # Generated evaluation plots (training_curve.png, beam_map.png, rate_cdf.png)
├── src/                  # Core source code
│   ├── __init__.py
│   ├── dataset.py        # Data loading, DFT codebook, and DataLoader construction
│   ├── model.py          # BeamPredictor Deep Residual MLP architecture
│   ├── train.py          # Training loop with AMP support and cosine annealing
│   └── evaluate.py       # Evaluation metrics (Top-k accuracy, achievable rate)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔒 Security & Best Practices

This codebase follows security best practices for ML pipelines:
- **Safe Checkpoint Loading**: Uses `torch.load(..., weights_only=True)` to prevent arbitrary code execution via malicious pickle payloads.
- **Input Validation**: Strict bounds checking on model hyperparameters loaded from checkpoints.
- **Path Sanitization**: Scenario names are validated against path traversal attacks before file operations.
- **Resource Guards**: Training configurations are validated to prevent OOM crashes or infinite loops.

## 📚 References

- **DeepMIMO Dataset**: [DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications](https://deepmimo.net/)
- **Beam Prediction**: Position-aided beam prediction for 6G massive MIMO systems.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
*Developed for VIT Chennai | BECE311L (Radar & Wireless Systems) | May 2026*
