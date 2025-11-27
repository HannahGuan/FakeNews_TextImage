# Baseline Multimodal Model: BERT + ResNet-50

This is a baseline model for multi-class classification that combines text and image features using concatenation fusion.

**Author:** Xinru Pan
**Date:** 2025-11-27

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
- `title`: Text content
- `image_url`: Path to image file (local path)
- `hasImage`: Boolean indicating if image exists (must be True for samples to be loaded)
- `2_way_label`: Binary label (0 or 1) for 2-way classification
- `3_way_label`: Label (0, 1, or 2) for 3-way classification (optional)
- `6_way_label`: Label (0-5) for 6-way classification (optional)

The dataset automatically filters out samples where `hasImage` is False or any required field is missing.

## Classification Types

The model supports three classification modes:

1. **2-way classification**: Real (0) vs. Fake (1)
2. **3-way classification**: Real (0), Fake with True Text (1), Fake with False Text (2)
3. **6-way classification**: Real (0), Satire/Parody (1), Misleading (2), Imposter (3), False Connection (4), Manipulated (5)

Set the classification type by changing the `LABEL_TYPE` parameter in the code.

## Training

Run the training script:

```bash
python baseline_model.py
```

### Hyperparameters

Default configuration:
- Batch size: 8
- Learning rate: 5e-4 (0.0005)
- Epochs: 10
- Hidden dimension: 512
- Dropout: 0.3
- Max sequence length: 128
- Label type: '2_way' (configurable to '3_way' or '6_way')

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

- **best_baseline_model.pth**: Best model checkpoint containing:
  - Epoch number
  - Model state dictionary
  - Optimizer state dictionary
  - Best dev accuracy

- Console output includes:
  - Training loss and accuracy per epoch
  - Validation (dev) loss and accuracy per epoch
  - Best model save notifications

## Model Performance

The script reports:
- Training loss and accuracy for each epoch
- Validation (dev) loss and accuracy for each epoch
- Best validation accuracy achieved
- Total and trainable parameter counts

## Image Handling

The model includes robust image loading:
- Supports local image paths
- Creates dummy black images if path doesn't exist
- Handles image loading errors gracefully with fallback to zero tensors
- Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

## Next Steps

Future improvements:
- Fine-tune BERT and/or ResNet-50 layers
- Experiment with different fusion strategies (attention mechanisms, gating)
- Add learning rate scheduling
- Implement early stopping
- Try different pretrained models (RoBERTa, ViT)
- Experiment with cross-modal attention
- Add data augmentation for images and text
- Implement weighted loss for class imbalance
