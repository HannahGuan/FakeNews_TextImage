# BLIP-2 Fake News Detection

## Architecture

### Pipeline Overview

```
Input: Image + Text
    ↓
BLIP-2 Feature Extractor (Pretrained, Frozen)
├─ Image Encoder (ViT-g)
├─ Text Encoder (Q-Former)
└─ Image-Text Matching Head
    ↓
Feature Vector [514 dimensions]
├─ Image Embeddings [256]
├─ Text Embeddings [256]
├─ Similarity Score [1]
└─ ITM Score [1]
    ↓
Classifier Head (Trainable)
├─ FC Layer 1: 514 → 512 + ReLU + Dropout
├─ FC Layer 2: 512 → 256 + ReLU + Dropout
└─ Output Layer: 256 → num_classes
    ↓
Softmax → Class Probabilities
```

### Key Components

**BLIP-2 Feature Extractor** (`blip2_extractor.py`): Extracts visual and semantic features from images and text. All weights are frozen for transfer learning.

**Classification Head** (`model.py`): Trainable MLP layers that take concatenated BLIP-2 features and output class predictions.

**Training Pipeline** (`train.py`): Handles data loading, training loop, validation, early stopping, and checkpoint saving.

**Evaluation Pipeline** (`evaluate.py`): Comprehensive metrics calculation, visualization generation, and detailed prediction logging.

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU support)
- 8GB+ RAM (16GB+ recommended)
- GPU with 8GB+ VRAM (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/HannahGuan/FakeNews_TextImage.git
cd blip2-fake-news-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
pillow>=9.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
```

## Usage

### Training

Basic training command:

```bash
python main.py --mode train --classification-type 2_way
```

Force CPU training:

```bash
python main.py --mode train --classification-type 2_way --force-cpu
```

### Evaluation

```bash
python main.py --mode eval --classification-type 2_way
```