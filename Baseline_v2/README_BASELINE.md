# Baseline Multimodal Model: BERT + ResNet-50

This is a baseline model for multi-class classification that combines text and image features using concatenation fusion.

**Author:** Xinru Pan
**Date:** 2025-12-01

## Steps

```bash
# Check class balance in your dataset
python check_class_balance.py

# Train baseline model
python baseline_model.py 

# Run grid search with fine-tuning 
python finetune_model.py
```

### Class Imbalance Analysis

Check your dataset balance before training:

```bash
python check_class_balance.py
```

This affects our choice of loss functions.

### Model Architecture

1. Text Encoder: BERT-base-uncased
   - Extracts CLS token embedding (768 dimensions)
   - Pretrained weights frozen during training

2. Image Encoder: ResNet-50
   - Extracts global average pooling output (2048 dimensions)
   - Pretrained weights frozen during training
   - Uses ImageNet pretrained weights

3. Fusion: Concatenation
   - Combined feature vector: 768 + 2048 = 2816 dimensions

4. Classifier: MLP with 2 Hidden Layers
   - Input: 2816 dimensions
   - First hidden layer: 512 dimensions with ReLU and Dropout (0.3)
   - Second hidden layer: 512 dimensions with ReLU and Dropout (0.3)
   - Output: N logits (2, 3, or 6 classes depending on classification type)

## Installation

```bash
pip install -r requirements.txt
```
**Files**
- Train CSV: `../train_sampled_with_images.csv`
- Dev CSV: `../dev_sampled_with_images.csv`
- Train images: `../train_images/{id}.jpg`
- Dev images: `../dev_images/{id}.jpg`

**Data Splitting:**
- The dev set is automatically split into validation (80%) and test (20%) using stratified sampling
- Split files are saved as `val_split_{label_type}.csv` and `test_split_{label_type}.csv`
The dataset automatically filters out samples where `hasImage` is False or any required field is missing.

## Classification Types

The model supports three classification modes:
1. 2-way classification: Real (0) vs. Fake (1)
2. 3-way classification: Real (0), Fake with True Text (1), Fake with False Text (2)
3. 6-way classification: Real (0), Satire/Parody (1), Misleading (2), Imposter (3), False Connection (4), Manipulated (5)

Set the classification type by changing the `LABEL_TYPE` parameter in the code.

## Training

Run the training script with command-line arguments:

```bash
# Basic usage (default: 2-way classification)
python baseline_model.py

# Specify classification type
python baseline_model.py --classification-type 3_way/6_way
```

### Hyperparameters

Default configuration:
- Batch size: 8
- Learning rate: 5e-4
- Epochs: 10
- Hidden dimension: 512
- Dropout: 0.3
- Max sequence length: 128

### Training Process

1. Only the MLP classifier is trained (BERT and ResNet-50 are frozen)
2. Uses Cross-Entropy Loss (supports multi-class classification)
3. Adam optimizer
4. Saves best model based on dev set accuracy

## Output

### Checkpoints
Saved to `checkpoints/{classification_type}/best_model.pth`

### Logs
Saved to `logs/{classification_type}/`

### Console Output
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Test loss and accuracy (evaluated at end)
- Dataset split sizes (train/val/test)
- Best model saved

## Model Performance

The script reports:
- Training loss and accuracy for each epoch
- Validation (dev) loss and accuracy for each epoch
- Best validation accuracy achieved
- Total and trainable parameter counts

## Evaluation

After training, use the companion evaluation script to get detailed metrics:

```bash
python evaluate_model.py
```

This will generate:
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix
- ROC curves (per-class, micro/macro averages for multi-class)
- Precision-Recall curves
- Confidence distribution plots
- Detailed prediction outputs with probabilities and logits

## Fine-Tuning with Grid Search

Use `finetune_model.py` for advanced hyperparameter tuning and handling class imbalance.

### Features

**Advanced Loss Functions:**
- **Focal Loss**: Addresses class imbalance by focusing on hard-to-classify examples (gamma=2.0)
- **Label Smoothing CrossEntropy**: Prevents overconfident predictions (smoothing=0.1)
- **Weighted CrossEntropy**: Uses inverse frequency class weights to handle imbalance
- **Standard CrossEntropy**: Baseline comparison

**Automatic Loss Selection by Classification Type:**
- **2-way classification**: Weighted CrossEntropy/Label Smoothing
- **3-way/6-way classification**: Focal Loss/Weighted CrossEntropy

**Hyperparameter Grid Search:**
- Learning rate: [1e-4, 5e-4, 1e-3]
- Batch size: [8, 16]
- Dropout: [0.3, 0.5]
- Pooling: [max, mean]
