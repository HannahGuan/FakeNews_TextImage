# Baseline Multimodal Model: BERT + ResNet-50

This is a baseline model for multi-class classification that combines text and image features using concatenation fusion.

**Author:** Xinru Pan
**Date:** 2025-11-27

## Quick Start

```bash
# Check class balance in your dataset
python check_class_balance.py

# Train baseline model (2-way classification)
python baseline_model.py --classification-type 2_way --epochs 10

# Run grid search with fine-tuning (automatically selects optimal loss functions)
python finetune_model.py --classification-type 2_way --epochs 10
```

## Table of Contents

- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Classification Types](#classification-types)
- [Training](#training)
- [Fine-Tuning with Grid Search](#fine-tuning-with-grid-search)
- [Evaluation](#evaluation)
- [Next Steps](#next-steps)

## Model Architecture

1. **Text Encoder**: BERT-base-uncased
   - Extracts CLS token embedding (768 dimensions)
   - Pretrained weights frozen during training

2. **Image Encoder**: ResNet-50
   - Extracts global average pooling output (2048 dimensions)
   - Pretrained weights frozen during training
   - Uses ImageNet pretrained weights

3. **Fusion**: Concatenation
   - Combined feature vector: 768 + 2048 = 2816 dimensions

4. **Classifier**: MLP with 2 Hidden Layers
   - Input: 2816 dimensions
   - First hidden layer: 512 dimensions with ReLU and Dropout (0.3)
   - Second hidden layer: 512 dimensions with ReLU and Dropout (0.3)
   - Output: N logits (2, 3, or 6 classes depending on classification type)

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- torch
- transformers
- torchvision
- pandas
- numpy
- tqdm
- Pillow
- certifi

## Dataset Format

The CSV files should contain the following columns:
- `clean_title`: Text content (cleaned title of the news article)
- `id`: Unique identifier used to load images (images stored as `{id}.jpg`)
- `hasImage`: Boolean indicating if image exists (must be True for samples to be loaded)
- `2_way_label`: Binary label (0 or 1) for 2-way classification
- `3_way_label`: Label (0, 1, or 2) for 3-way classification
- `6_way_label`: Label (0-5) for 6-way classification

**Directory Structure:**
```
Data/
├── Baseline/
│   ├── baseline_model.py
│   ├── finetune_model.py
│   ├── check_class_balance.py
│   └── README_BASELINE.md
├── train_sampled_with_images.csv
├── dev_sampled_with_images.csv
├── train_images/
│   └── {id}.jpg
└── dev_images/
    └── {id}.jpg
```

- Train CSV: `../train_sampled_with_images.csv`
- Dev CSV: `../dev_sampled_with_images.csv`
- Train images: `../train_images/{id}.jpg`
- Dev images: `../dev_images/{id}.jpg`

**Data Splitting:**
- The dev set is automatically split into validation (80%) and test (20%) using stratified sampling
- Split files are saved for reproducibility as `val_split_{label_type}.csv` and `test_split_{label_type}.csv`

The dataset automatically filters out samples where `hasImage` is False or any required field is missing.

## Classification Types

The model supports three classification modes:

1. **2-way classification**: Real (0) vs. Fake (1)
2. **3-way classification**: Real (0), Fake with True Text (1), Fake with False Text (2)
3. **6-way classification**: Real (0), Satire/Parody (1), Misleading (2), Imposter (3), False Connection (4), Manipulated (5)

Set the classification type by changing the `LABEL_TYPE` parameter in the code.

## Training

Run the training script with command-line arguments:

```bash
# Basic usage (default: 2-way classification)
python baseline_model.py

# Specify classification type
python baseline_model.py --classification-type 3_way

# Customize hyperparameters
python baseline_model.py --batch-size 16 --epochs 20 --lr 1e-3

# Force CPU usage (if GPU available but want to use CPU)
python baseline_model.py --force-cpu
```

### Command-Line Arguments

- `--classification-type`: Classification mode (`2_way`, `3_way`, or `6_way`) - Default: `2_way`
- `--batch-size`: Batch size for training - Default: `8`
- `--epochs`: Number of training epochs - Default: `10`
- `--lr`: Learning rate - Default: `5e-4` (0.0005)
- `--force-cpu`: Force CPU usage even if GPU available

### Hyperparameters

Default configuration:
- Batch size: 8 (configurable via `--batch-size`)
- Learning rate: 5e-4 (configurable via `--lr`)
- Epochs: 10 (configurable via `--epochs`)
- Hidden dimension: 512 (fixed in code)
- Dropout: 0.3 (fixed in code)
- Max sequence length: 128 (fixed in code)
- Label type: '2_way' (configurable via `--classification-type`)

### Training Process

1. Only the MLP classifier is trained (BERT and ResNet-50 are frozen)
2. Uses Cross-Entropy Loss (supports multi-class classification)
3. Adam optimizer (only optimizes classifier parameters)
4. Saves best model based on dev set accuracy
5. Progress bars with tqdm for training and evaluation

### Model Parameters

For the default configuration:
- Total parameters: ~93M (most from frozen BERT and ResNet-50)
- Trainable parameters: ~1.8M (only the MLP classifier)

## Output

### Checkpoints
Saved to `checkpoints/{classification_type}/best_model.pth`:
- Epoch number
- Model state dictionary
- Optimizer state dictionary
- Validation accuracy and loss
- Label type

### Logs
Saved to `logs/{classification_type}/`:
- `training_history_{classification_type}.png`: Training curves showing:
  - Training and validation loss
  - Training and validation accuracy
  - Detailed validation accuracy plot

### Console Output
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Test loss and accuracy (evaluated at end)
- Dataset split sizes (train/val/test)
- Best model save notifications

## Model Performance

The script reports:
- Training loss and accuracy for each epoch
- Validation (dev) loss and accuracy for each epoch
- Best validation accuracy achieved
- Total and trainable parameter counts

## Image Handling

The model includes robust image loading:
- Images loaded from `{image_dir}/{id}.jpg` using the `id` column from CSV
- Automatic conversion to RGB format
- Handles missing images gracefully with fallback to zero tensors
- Prints warnings for failed image loads
- Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Separate directories for train and dev images

## Evaluation

After training, use the companion evaluation script to get detailed metrics:

```bash
python evaluate_model.py
```

This will generate:
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix visualization
- ROC curves (per-class, micro/macro averages for multi-class)
- Precision-Recall curves
- Confidence distribution plots
- Detailed prediction outputs with probabilities and logits

## Fine-Tuning with Grid Search

The project includes `finetune_model.py` for advanced hyperparameter tuning and handling class imbalance.

### Features

**Advanced Loss Functions:**
- **Focal Loss**: Addresses class imbalance by focusing on hard-to-classify examples (gamma=2.0)
- **Label Smoothing CrossEntropy**: Prevents overconfident predictions (smoothing=0.1)
- **Weighted CrossEntropy**: Uses inverse frequency class weights to handle imbalance
- **Standard CrossEntropy**: Baseline comparison

**Automatic Loss Selection by Classification Type:**
- **2-way classification**: Uses Weighted CrossEntropy and Label Smoothing
- **3-way/6-way classification**: Uses Focal Loss and Weighted CrossEntropy

**Hyperparameter Grid Search:**
- Learning rate: [1e-4, 5e-4, 1e-3]
- Batch size: [8, 16]
- Hidden dimensions: [256, 512, 768]
- Dropout: [0.2, 0.3, 0.4]
- Weight decay: [0.0, 1e-5, 1e-4]
- Learning rate scheduling: [False, True]

### Class Imbalance Analysis

Check your dataset balance before training:

```bash
python check_class_balance.py
```

**Results from sample dataset:**
- **2-way**: 59.16% vs 40.84% - Relatively balanced ✓
- **3-way**: Highly imbalanced (minority class: 2.33%) ⚠️
- **6-way**: Highly imbalanced (minority class: 1.72%) ⚠️

### Usage

```bash
# Basic usage (2-way classification, 10 epochs)
python finetune_model.py

# 3-way classification with 15 epochs
python finetune_model.py --classification-type 3_way --epochs 15

# 6-way classification without class weights
python finetune_model.py --classification-type 6_way --no-class-weights

# Force CPU usage
python finetune_model.py --force-cpu
```

### Command-Line Arguments

- `--classification-type`: Choose `2_way`, `3_way`, or `6_way` (Default: `2_way`)
- `--epochs`: Number of training epochs per configuration (Default: `10`)
- `--use-class-weights`: Enable automatic class weight calculation (Default: `True`)
- `--no-class-weights`: Disable class weights
- `--force-cpu`: Force CPU usage even if GPU available

### Output Files

**Grid Search Results** (`grid_search_results/`):
- `best_model_grid_search.pth`: Best performing model checkpoint
- `grid_search_results_<timestamp>.json`: Detailed results for all configurations
- `summary_<timestamp>.txt`: Human-readable summary report with top 10 configurations

**Validation Splits**:
- `val_split_{label_type}_finetune.csv`: Validation set
- `test_split_{label_type}_finetune.csv`: Test set

### Grid Search Process

The script automatically:
1. Computes class weights based on training data distribution
2. Selects appropriate loss functions for the classification type
3. Tests all hyperparameter combinations
4. Tracks the best performing model based on validation accuracy
5. Saves detailed results and summary reports

**Example Output:**
```
Total configurations to test: 216
[1/216] Testing configuration:
  loss_type: weighted_ce
  learning_rate: 0.0001
  batch_size: 8
  ...

*** NEW BEST MODEL: 87.45% ***
```

### Performance Comparison

The grid search helps identify optimal configurations for each classification type:
- Automatically handles class imbalance through weighted losses
- Finds best learning rate and regularization settings
- Compares different architectural choices (hidden dimensions, dropout)

## Next Steps

Future improvements:
- Fine-tune BERT and/or ResNet-50 layers
- Experiment with different fusion strategies (attention mechanisms, gating)
- Try different pretrained models (RoBERTa, ViT)
- Experiment with cross-modal attention
- Add data augmentation for images and text
