"""
Model Evaluation and Visualization
Generates comprehensive performance metrics and plots.

Author: Xinru Pan
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score
)
from tqdm import tqdm
import json
from baseline_model import MultiModalClassifier, MultiModalDataset
from transformers import BertTokenizer
from torchvision import transforms


# set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def evaluate_model_detailed(model, dataloader, device):
    """
    Evaluate model and collect predictions for detailed metrics.

    Returns:
        dict: Contains labels, predictions, probabilities, and logits
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_logits = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask, images).squeeze(1)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()

            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    return {
        'labels': np.array(all_labels),
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'logits': np.array(all_logits)
    }



def plot_roc_curve(labels, probabilities, save_path='roc_curve.png'):
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to {save_path}")
    plt.close()

    return roc_auc


def plot_precision_recall_curve(labels, probabilities, save_path='precision_recall_curve.png'):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    avg_precision = average_precision_score(labels, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Precision-Recall curve to {save_path}")
    plt.close()

    return avg_precision


def generate_classification_report(labels, predictions, probabilities):
    """Print core classification metrics (Accuracy, F1, ROC-AUC, Avg Precision)."""
    accuracy = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions)
    roc_auc = auc(*roc_curve(labels, probabilities)[:2])
    avg_precision = average_precision_score(labels, probabilities)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

def main():
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    MODEL_PATH = 'best_baseline_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dev_dataset = MultiModalDataset(
        'dev_sampled_with_images.csv',
        tokenizer,
        image_transform,
        max_length=MAX_LENGTH
    )
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = MultiModalClassifier(hidden_dim=512, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"\nLoaded model from epoch {checkpoint['epoch']} "
          f"(dev acc: {checkpoint['dev_acc']:.2f}%)")

    results = evaluate_model_detailed(model, dev_loader, device)

    plot_roc_curve(results['labels'], results['probabilities'])
    plot_precision_recall_curve(results['labels'], results['probabilities'])

    print("\nKey Metrics:")
    metrics = generate_classification_report(
        results['labels'],
        results['predictions'],
        results['probabilities']
    )
    print(f"\nAccuracy:          {metrics['accuracy']:.4f}")
    print(f"F1 Score:          {metrics['f1_score']:.4f}")
    print(f"ROC AUC:           {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['avg_precision']:.4f}")


if __name__ == '__main__':
    main()
