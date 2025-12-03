# Model Evaluation Guide

**Author:** Xinru Pan

## Overview

The evaluation script loads a trained model checkpoint and generates comprehensive performance metrics, visualizations, and detailed predictions for test data. It supports 2-way, 3-way, and 6-way classification tasks.

### What This Guide Covers

- Command-line usage and arguments
- Understanding all output files and metrics
- Interpreting visualizations
- Error analysis workflows
- Model comparison strategies
- Troubleshooting common issues

## Quick Start

**Basic evaluation:**
```bash
python evaluate_model.py \
  --checkpoint-path grid_search_results/best_model_grid_search.pth \
  --classification-type 2_way
```

This will:
1. Load the trained model from the checkpoint
2. Evaluate on the default test dataset
3. Generate all metrics and visualizations
4. Save results to `evaluation_results/` directory

## Installation

Required packages:
```bash
pip install torch transformers torchvision pandas numpy scikit-learn matplotlib seaborn tqdm
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint-path` | str | `grid_search_results/best_model_grid_search.pth` | Path to trained model checkpoint (.pth file) |
| `--classification-type` | str | `2_way` | Classification type: `2_way`, `3_way`, or `6_way` |
| `--test-csv` | str | `../test_split_2_way_finetune.csv` | Path to test dataset CSV file |
| `--test-images-dir` | str | `../dev_images` | Path to test images directory |
| `--output-dir` | str | `evaluation_results` | Directory to save evaluation results |
| `--batch-size` | int | `16` | Batch size for evaluation |
| `--max-length` | int | `128` | Maximum sequence length for text tokenization |
| `--num-workers` | int | `2` | Number of data loading workers |
| `--force-cpu` | flag | - | Force CPU usage (disable GPU) |

## Usage Examples

### 2-way Classification

```bash
python evaluate_model.py \
  --checkpoint-path grid_search_results/best_model_grid_search.pth \
  --classification-type 2_way \
  --test-csv ../test_split_2_way_finetune.csv \
  --output-dir evaluation_results_2way
```

### 3-way Classification

```bash
python evaluate_model.py \
  --checkpoint-path grid_search_results/best_model_3way.pth \
  --classification-type 3_way \
  --test-csv ../test_split_3_way_finetune.csv \
  --output-dir evaluation_results_3way
```

### 6-way Classification

```bash
python evaluate_model.py \
  --checkpoint-path grid_search_results/best_model_6way.pth \
  --classification-type 6_way \
  --test-csv ../test_split_6_way_finetune.csv \
  --output-dir evaluation_results_6way
```

### Custom Test Dataset

```bash
python evaluate_model.py \
  --checkpoint-path my_models/custom_model.pth \
  --classification-type 2_way \
  --test-csv /path/to/custom_test.csv \
  --test-images-dir /path/to/custom_images \
  --batch-size 32
```

### CPU-only Evaluation

```bash
python evaluate_model.py \
  --checkpoint-path model.pth \
  --classification-type 2_way \
  --force-cpu
```

## Checkpoint File Format

The script expects checkpoint files saved by [finetune_model.py](finetune_model.py) with the following structure:

```python
{
    'config': {
        'num_classes': int,        # 2, 3, or 6
        'hidden_dim': int,         # e.g., 512
        'dropout': float,          # e.g., 0.3
        'pooling_strategy': str,   # 'mean', 'max', or 'cls'
        'loss_type': str,          # e.g., 'weighted_ce', 'focal'
        'learning_rate': float,
        'batch_size': int,
        ...
    },
    'model_state_dict': OrderedDict(...),  # Model weights
    'dev_acc': float,              # Validation accuracy (optional)
}
```

## Generated Outputs

All outputs are saved to the directory specified by `--output-dir` (default: `evaluation_results/`).

### 1. Metrics Summary

**File:** `metrics_{classification_type}.json`

Contains overall and per-class metrics:
```json
{
  "accuracy": 0.7845,
  "precision": 0.7823,
  "recall": 0.7845,
  "f1": 0.7812,
  "roc_auc": 0.8523,
  "average_confidence": 0.8234,
  "average_entropy": 0.3421,
  "per_class": {
    "Real": {
      "precision": 0.8012,
      "recall": 0.7845,
      "f1": 0.7923
    },
    "Fake": {
      "precision": 0.7634,
      "recall": 0.7845,
      "f1": 0.7701
    }
  }
}
```

**Metrics Explained:**
- **Accuracy**: Overall percentage correct
- **Precision (weighted)**: Average precision across classes, weighted by support
- **Recall (weighted)**: Average recall across classes, weighted by support
- **F1 (weighted)**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall discriminative ability (multi-class uses one-vs-rest)
- **Average Confidence**: Mean of maximum class probabilities
- **Average Entropy**: Mean prediction uncertainty (lower = more confident)

### 2. Detailed Predictions

**File:** `predictions_{classification_type}.json`

Individual predictions with all probabilities and uncertainty metrics:
```json
[
  {
    "sample_id": 0,
    "true_label": 1,
    "true_class_name": "Fake",
    "predicted_label": 1,
    "predicted_class_name": "Fake",
    "correct": true,
    "confidence": 0.9234,
    "true_class_probability": 0.9234,
    "all_probabilities": {
      "Real": 0.0766,
      "Fake": 0.9234
    },
    "all_logits": {
      "Real": -2.134,
      "Fake": 2.456
    },
    "prediction_entropy": 0.2341,
    "max_probability": 0.9234,
    "probability_margin": 0.8468,
    "error_type": null
  }
]
```

**Fields Explained:**
- **sample_id**: Sample index in test set
- **confidence**: Model's confidence in predicted class
- **true_class_probability**: Model's probability for true class
- **all_probabilities**: Probabilities for all classes
- **all_logits**: Raw logits for all classes
- **prediction_entropy**: Entropy of prediction distribution (higher = more uncertain)
- **max_probability**: Maximum probability across classes
- **probability_margin**: Difference between top 2 probabilities (higher = more decisive)
- **error_type**: Description of error (null if correct)

### 3. Confusion Matrix

**Files:**
- `confusion_matrix_{classification_type}.png` - Heatmap visualization
- `confusion_matrix_{classification_type}.json` - Raw data

**PNG:** Heatmap with actual counts showing where predictions were correct/incorrect

**JSON:**
```json
{
  "confusion_matrix": [[TN, FP], [FN, TP]],
  "class_names": ["Real", "Fake"],
  "normalized": false
}
```

**How to interpret:**
- **Diagonal cells**: Correct predictions (should be high)
- **Off-diagonal cells**: Misclassifications (should be low)
- **Row i, Column j**: Number of samples from class i predicted as class j
- Darker blue colors indicate higher counts

### 4. ROC Curves

**Files:**
- `roc_curve_{classification_type}.png` - Visualization
- `roc_data_{classification_type}.json` - FPR, TPR, thresholds, AUC values

**Binary Classification (2-way):**
- Single ROC curve with one AUC score
- AUC interpretation:
  - 0.5: Random classifier
  - 0.7-0.8: Acceptable
  - 0.8-0.9: Good
  - 0.9+: Excellent

**Multi-class (3-way, 6-way):**
- **Per-class curves**: One-vs-rest ROC for each class
- **Micro-average**: Aggregate all classes (good for imbalanced datasets)
- **Macro-average**: Average of per-class AUCs (treats all classes equally)

**JSON structure (multi-class):**
```json
{
  "classification_type": "multi_class",
  "class_names": ["Real", "Fake (True Text)", "Fake (False Text)"],
  "per_class": {
    "Real": {
      "fpr": [0.0, 0.05, 0.1, ...],
      "tpr": [0.0, 0.3, 0.5, ...],
      "thresholds": [1.0, 0.95, 0.9, ...],
      "auc": 0.85
    }
  },
  "micro_average": {"fpr": [...], "tpr": [...], "auc": 0.82},
  "macro_average": {"fpr": [...], "tpr": [...], "auc": 0.81}
}
```

**How to interpret:**
- Curves closer to top-left corner = better performance
- Compare per-class AUCs to identify easier/harder classes
- Micro-average AUC gives overall multi-class performance

### 5. Precision-Recall Curves

**Files:**
- `pr_curve_{classification_type}.png` - Visualization
- `pr_data_{classification_type}.json` - Precision, recall, thresholds, AP values

**Key Metrics:**
- **Average Precision (AP)**: Area under PR curve
- **Precision**: Of predicted positives, how many are correct?
- **Recall**: Of actual positives, how many did we find?

**JSON structure:**
```json
{
  "classification_type": "binary",
  "class_names": ["Real", "Fake"],
  "precision": [1.0, 0.95, 0.9, ...],
  "recall": [0.0, 0.1, 0.2, ...],
  "thresholds": [0.9, 0.85, 0.8, ...],
  "average_precision": 0.87
}
```

**When to use PR curves vs ROC:**
- PR curves are better for imbalanced datasets
- ROC curves give overall discriminative ability
- Both provide complementary information

### 6. Confidence Distribution

**File:** `confidence_distribution_{classification_type}.png`

Histogram comparing confidence (max probability) for correct vs incorrect predictions.

**Shows:**
- Green histogram: Confidence scores for correct predictions
- Red histogram: Confidence scores for incorrect predictions

**How to interpret:**
- **Well-calibrated model**: Correct predictions have high confidence, incorrect ones have low confidence
- **Overconfident model**: Both distributions shifted right (high confidence even when wrong)
- **Underconfident model**: Both distributions shifted left (low confidence even when correct)
- **Ideal**: Clear separation with green (right) and red (left)

## Classification Types

### 2-way Classification
- Class 0: Real
- Class 1: Fake

**Key Metrics:**
- Accuracy: >80% acceptable, >85% good, >90% excellent
- F1-Score: 0.8+ is good
- ROC-AUC: >0.85 good, >0.90 excellent

### 3-way Classification
- Class 0: Real
- Class 1: Fake (True Text)
- Class 2: Fake (False Text)

**Considerations:**
- More classes = harder task
- Accuracy typically lower than 2-way
- Focus on per-class F1 scores
- Use confusion matrix to identify confused classes

### 6-way Classification
- Class 0: Real
- Class 1: Satire/Parody
- Class 2: Misleading
- Class 3: Imposter
- Class 4: False Connection
- Class 5: Manipulated

**Most Challenging:**
- Accuracy >60% reasonable, >70% good
- Per-class performance varies significantly
- Some classes may be very similar
- Check confusion matrix for systematic confusions

## Console Output Example

```
Using device: cuda
Classification type: 2_way (2 classes)

Loading checkpoint from grid_search_results/best_model_grid_search.pth
Model loaded successfully!
Configuration: {'loss_type': 'weighted_ce', 'learning_rate': 0.0001, ...}
Checkpoint Dev Accuracy: 78.45%

Loading test dataset from ../test_split_2_way_finetune.csv
Test dataset size: 1000

Evaluating model on 63 batches...
Evaluating: 100%|████████████████| 63/63 [00:15<00:00, 4.12it/s]

======================================================================
EVALUATION RESULTS
======================================================================
Classification Type: 2_way
Number of Classes: 2
Total Samples: 1000
----------------------------------------------------------------------
ACCURACY: 0.7845
PRECISION: 0.7823
RECALL: 0.7845
F1: 0.7812
ROC_AUC: 0.8523
AVERAGE_CONFIDENCE: 0.8234
AVERAGE_ENTROPY: 0.3421

----------------------------------------------------------------------
Per-Class Metrics:
----------------------------------------------------------------------

Real:
  precision: 0.8012
  recall: 0.7845
  f1: 0.7923

Fake:
  precision: 0.7634
  recall: 0.7845
  f1: 0.7701

----------------------------------------------------------------------
Classification Report:
----------------------------------------------------------------------
              precision    recall  f1-score   support

        Real       0.80      0.78      0.79       500
        Fake       0.76      0.78      0.77       500

    accuracy                           0.78      1000
   macro avg       0.78      0.78      0.78      1000
weighted avg       0.78      0.78      0.78      1000

[SAVE] Metrics saved to evaluation_results/metrics_2_way.json
[SAVE] Predictions saved to evaluation_results/predictions_2_way.json
...
======================================================================
EVALUATION COMPLETE
======================================================================
Results saved to: evaluation_results
Final Test Accuracy: 0.7845
```

## Error Analysis Workflow

### 1. Load Predictions

```python
import json
import pandas as pd

with open('evaluation_results/predictions_2_way.json') as f:
    predictions = json.load(f)

df = pd.DataFrame(predictions)
```

### 2. Find Incorrect Predictions

```python
errors = df[df['correct'] == False]
print(f"Total errors: {len(errors)}")
```

### 3. Analyze Low-Confidence Errors

```python
low_conf_errors = errors[errors['confidence'] < 0.7]
print(f"Low confidence errors: {len(low_conf_errors)}")
```

### 4. Analyze High-Confidence Errors

```python
high_conf_errors = errors[errors['confidence'] > 0.9]
print(f"High confidence errors (model very wrong): {len(high_conf_errors)}")
```

### 5. Group by Error Type

```python
from collections import Counter

error_types = Counter(errors['error_type'])
print(error_types)
# {'predicted_Fake_actual_Real': 112, 'predicted_Real_actual_Fake': 103}
```

### 6. Find Most Uncertain Predictions

```python
uncertain = df.nlargest(10, 'prediction_entropy')
print(uncertain[['sample_id', 'prediction_entropy', 'correct']])
```

### 7. Analyze Probability Margins

```python
close_calls = df[df['probability_margin'] < 0.2]
print(f"Close predictions: {len(close_calls)}")
print(f"Accuracy on close calls: {close_calls['correct'].mean():.4f}")
```

## Model Comparison

To compare multiple models:

```bash
# Evaluate Model 1
python evaluate_model.py \
  --checkpoint-path model1.pth \
  --classification-type 2_way \
  --output-dir results_model1

# Evaluate Model 2
python evaluate_model.py \
  --checkpoint-path model2.pth \
  --classification-type 2_way \
  --output-dir results_model2
```

Then compare metrics:
```python
import json

with open('results_model1/metrics_2_way.json') as f:
    metrics1 = json.load(f)

with open('results_model2/metrics_2_way.json') as f:
    metrics2 = json.load(f)

print("Model Comparison:")
print(f"Model 1 - Accuracy: {metrics1['accuracy']:.4f}, F1: {metrics1['f1']:.4f}")
print(f"Model 2 - Accuracy: {metrics2['accuracy']:.4f}, F1: {metrics2['f1']:.4f}")
```

## Performance Diagnostics

### Signs of Overfitting

- Large gap between train and validation accuracy
- Training loss decreases but validation loss increases
- High confidence on training data, low confidence on validation

**Solutions:**
- Increase dropout (try 0.4-0.5)
- Add data augmentation
- Use early stopping
- Reduce model complexity

### Signs of Underfitting

- Both train and validation accuracy are low
- Model predictions close to random
- High entropy across all predictions

**Solutions:**
- Fine-tune BERT/ResNet (unfreeze layers)
- Increase model capacity (larger hidden dimensions)
- Train for more epochs
- Try different learning rates

### Class Imbalance Issues

**Symptoms:**
- High accuracy but poor recall for minority classes
- Model predicts majority class most of the time
- Large difference between per-class F1 scores

**Solutions:**
- Use weighted loss function (already in finetune_model.py)
- Oversample minority classes
- Use focal loss for hard examples
- Focus on F1-score rather than accuracy

### Poor Calibration

**Symptoms:**
- Good accuracy but poor confidence distribution separation
- High confidence on incorrect predictions
- ROC-AUC is good but probability estimates unreliable

**Solutions:**
- Apply temperature scaling after training
- Use label smoothing (available in finetune_model.py)
- Add calibration loss during training

## Threshold Optimization

For applications requiring specific precision/recall trade-offs:

```python
import json

# Load PR data
with open('evaluation_results/pr_data_2_way.json') as f:
    pr_data = json.load(f)

precision = pr_data['precision']
recall = pr_data['recall']
thresholds = pr_data['thresholds']

# Find threshold for 90% precision
target_precision = 0.90
idx = next(i for i, p in enumerate(precision) if p >= target_precision)
optimal_threshold = thresholds[idx]
corresponding_recall = recall[idx]

print(f"For {target_precision:.1%} precision:")
print(f"  Use threshold: {optimal_threshold:.4f}")
print(f"  Expected recall: {corresponding_recall:.4f}")
```

## Troubleshooting

### Issue: Checkpoint not found
```
FileNotFoundError: Checkpoint not found: grid_search_results/best_model_grid_search.pth
```
**Solution:** Verify the checkpoint path. Use absolute or relative path from script directory.

### Issue: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce batch size: `--batch-size 8`
- Use CPU: `--force-cpu`

### Issue: Mismatched classification type
```
KeyError: 'The label column "3_way_label" is not in the dataset'
```
**Solution:** Ensure `--classification-type` matches the labels in your test CSV.

### Issue: Missing images
```
FileNotFoundError: Image not found: /path/to/image.jpg
```
**Solution:** Verify `--test-images-dir` points to correct directory with images.

### Issue: Model config mismatch
```
RuntimeError: Error loading state_dict
```
**Solution:** Checkpoint was saved from different model architecture. Ensure checkpoint matches `EnhancedMultiModalClassifier`.

## Performance Tips

1. **Use GPU:** Evaluation is 10-20× faster on GPU
2. **Increase batch size:** Larger batches utilize GPU better
3. **Adjust num_workers:** More workers speed up data loading
4. **Pre-extract features:** For multiple evaluations, consider caching features

## Integration with Fine-tuning

### After Fine-tuning

```bash
# 1. Fine-tune model
python finetune_model.py --classification-type 2_way --epochs 10

# 2. Evaluate best model from grid search
python evaluate_model.py \
  --checkpoint-path grid_search_results/best_model_grid_search.pth \
  --classification-type 2_way
```

### Custom Python Integration

```python
from evaluate_model import ModelEvaluator, load_model_checkpoint
from pathlib import Path
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, config = load_model_checkpoint('model.pth', device)

# Create evaluator
evaluator = ModelEvaluator(
    model=model,
    num_classes=2,
    classification_type='2_way',
    device=device,
    output_dir=Path('my_results')
)

# Evaluate on custom dataloader
metrics, labels, preds, probs, logits = evaluator.evaluate(my_dataloader)
```

## Best Practices

### During Development

1. Start with 2-way classification first
2. Check baselines: compare against random and majority-class
3. Use small dev set for fast iteration
4. Monitor per-class metrics, not just overall accuracy

### Before Deployment

1. Final evaluation on held-out test set
2. Cross-validation with different seeds
3. Threshold tuning on validation set
4. Verify confidence scores are meaningful
5. Understand failure modes through error analysis

### Reporting Results

Include in your report:
- Confusion matrix visualization
- ROC-AUC and PR-AUC scores
- Per-class F1 scores
- Confidence distribution plot
- Example misclassifications with analysis

## Related Files

- Fine-tuning script: [finetune_model.py](finetune_model.py)
- Model architecture: [finetune_model.py](finetune_model.py:33-122) (EnhancedMultiModalClassifier)
- Dataset loader: [baseline_model.py](baseline_model.py) (MultiModalDataset)
- Fine-tuning README: [README_finetune_model.md](README_finetune_model.md)

## Citation

```
Author: Xinru Pan
Course: CS230 (Stanford)
Purpose: Multimodal Classification Evaluation
```
