# Fine-Tuning Multimodal Classifier with Grid Search

**Author:** Xinru Pan

A comprehensive fine-tuning script for multimodal classification combining BERT and ResNet-50 with extensive hyperparameter grid search capabilities.

## Overview

This script performs fine-tuning and hyperparameter optimization for a multimodal classifier that combines:
- **Text Encoder:** BERT-base-uncased (pretrained, frozen)
- **Image Encoder:** ResNet-50 (pretrained, frozen)
- **Classifier:** 2-layer MLP with ReLU activation and dropout

The script supports multiple classification types (2-way, 3-way, 6-way), various loss functions, pooling strategies, and automated grid search for hyperparameter tuning.

## Features

### Model Architecture

**EnhancedMultiModalClassifier**
- BERT text encoder with configurable pooling strategies (CLS, mean, max)
- ResNet-50 image encoder with global average pooling
- Feature concatenation (768-dim BERT + 2048-dim ResNet = 2816-dim)
- 2-layer MLP classifier with configurable hidden dimensions and dropout

### Loss Functions

1. **Cross Entropy (CE):** Standard cross-entropy loss
2. **Weighted Cross Entropy:** CE with inverse frequency class weights
3. **Focal Loss:** Addresses class imbalance with focusing parameter
   - Formula: `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`
   - Focuses on hard-to-classify examples
4. **Label Smoothing:** Prevents overconfident predictions
   - Softens hard labels: `[0, 0, 1, 0]` → `[0.025, 0.025, 0.925, 0.025]`

### Pooling Strategies

- **CLS:** Use BERT's [CLS] token (first token)
- **Mean:** Mean pooling over sequence with attention masking
- **Max:** Max pooling over sequence with attention masking

### Grid Search Capabilities

Automated hyperparameter search over:
- Learning rates
- Batch sizes
- Dropout rates
- Loss functions
- Pooling strategies
- Hidden dimensions
- Focal loss gamma values
- Label smoothing factors

## Requirements

```bash
torch
transformers
torchvision
pandas
numpy
tqdm
scikit-learn
certifi
```

## Usage

### Basic Usage

```bash
python finetune_model.py --classification-type 2_way
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--classification-type` | str | `2_way` | Classification type: `2_way`, `3_way`, or `6_way` |
| `--epochs` | int | `10` | Number of training epochs per configuration |
| `--use-class-weights` | flag | `True` | Use inverse frequency class weights |
| `--no-class-weights` | flag | - | Disable class weights |
| `--force-cpu` | flag | - | Force CPU usage (disable GPU) |

### Examples

**2-way classification with class weights:**
```bash
python finetune_model.py --classification-type 2_way --epochs 10
```

**3-way classification without class weights:**
```bash
python finetune_model.py --classification-type 3_way --no-class-weights --epochs 15
```

**6-way classification on CPU:**
```bash
python finetune_model.py --classification-type 6_way --force-cpu
```

## Data Requirements

### Expected Directory Structure

```
Data/
├── Baseline/
│   └── finetune_model.py (this script)
├── train_sampled_with_images.csv
├── dev_sampled_with_images.csv
├── train_images/
│   └── [image files]
└── dev_images/
    └── [image files]
```

### CSV Format

CSV files should contain:
- `text`: Text content
- `image_filename`: Image file name
- `2_way_label`, `3_way_label`, `6_way_label`: Classification labels

## Parameter Grid Configuration

### 2-way Classification
- Loss functions: Weighted CE, Label Smoothing
- Learning rates: [1e-4, 5e-4, 5e-5]
- Batch sizes: [8, 16]
- Dropout: [0.3, 0.5]
- Pooling: [max, mean]

### 3-way and 6-way Classification
- Loss functions: Weighted CE, Focal Loss
- Learning rates: [1e-4, 5e-4, 5e-5]
- Batch sizes: [8, 16]
- Dropout: [0.3, 0.5]
- Pooling: [max, mean]

**Total configurations tested:** 24 per classification type

## Output

### Generated Files

**Saved to `grid_search_results/` directory:**

1. **`best_model_grid_search.pth`:** Best model checkpoint
   - Contains: config, model_state_dict, dev_acc

2. **`grid_search_results_{timestamp}.json`:** Detailed results for all configurations
   - Configuration parameters
   - Training history (loss, accuracy per epoch)
   - Best epoch and accuracy

3. **`summary_{timestamp}.txt`:** Human-readable summary report
   - Top 10 configurations ranked by validation accuracy
   - Success/failure statistics
   - Configuration details

4. **`val_split_{classification_type}_finetune.csv`:** Validation split
5. **`test_split_{classification_type}_finetune.csv`:** Test split (80/20 split of dev set)

### Console Output

During training, the script displays:
- Device information (CPU/GPU)
- Class distribution and weights
- Parameter grid summary
- Training progress with loss and accuracy
- Best model updates
- Final summary with best configuration


## Key Functions

### Training Functions

- **`train_epoch()`:** Train model for one epoch
- **`evaluate()`:** Evaluate on validation/test set
- **`train_single_config()`:** Train with single hyperparameter configuration

### Loss Functions

- **`FocalLoss`:** Focal loss implementation
- **`LabelSmoothingCrossEntropy`:** Label smoothing implementation
- **`get_loss_function()`:** Factory function for loss selection

### Utilities

- **`compute_class_weights()`:** Calculate inverse frequency weights
- **`generate_summary_report()`:** Generate human-readable summary
- **`grid_search()`:** Perform hyperparameter grid search

## Class Weights

When enabled (`--use-class-weights`), the script automatically computes inverse frequency weights:

```
weight_i = total_samples / (num_classes * class_i_samples)
```

This helps handle class imbalance by upweighting minority classes.

## Model Architecture Details

```
Input: (text, image)
  ↓
[BERT-base] → [768-dim] ──┐
                          ├→ [2816-dim] → MLP → [num_classes]
[ResNet-50] → [2048-dim] ─┘

MLP:
  Linear(2816, 512) → ReLU → Dropout
  Linear(512, 512) → ReLU → Dropout
  Linear(512, num_classes)
```

## Training Strategy

1. **Frozen Encoders:** BERT and ResNet-50 parameters are frozen
2. **Trainable Classifier:** Only MLP classifier is trained
3. **Optimizer:** Adam with configurable learning rate and weight decay
4. **Scheduler:** Optional ReduceLROnPlateau (currently disabled)
5. **Early Stopping:** Best model tracked by validation accuracy

## Performance Tips

- Use GPU for faster training (automatically detected)
- Adjust batch size based on GPU memory
- Reduce parameter grid for faster experiments
- Monitor class distribution for severe imbalance
- Consider label smoothing for 2-way classification
- Use focal loss for multi-class imbalanced datasets

## Dependencies on Other Files

- **`baseline_model.py`:** Imports `MultiModalDataset` class
  - Handles data loading and preprocessing
  - Tokenizes text with BERT tokenizer
  - Loads and transforms images

## Notes

- Script includes SSL certificate handling for secure downloads
- Automatically splits dev set into validation (80%) and test (20%)
- Preserves stratification during splits
- Includes error handling for failed configurations
- Saves results incrementally to prevent data loss

## Troubleshooting

**CUDA out of memory:**
- Reduce batch size in parameter grid
- Use `--force-cpu` flag

**Import errors:**
- Ensure `baseline_model.py` is in same directory
- Install all required packages

**Large grid search:**
- Script warns if >50 configurations
- Consider reducing parameter ranges
- Use random search for very large grids

## Citation

If you use this code, please cite:
```
Author: Xinru Pan
Course: CS230 (Stanford)
Purpose: Multimodal Classification with Grid Search
```
