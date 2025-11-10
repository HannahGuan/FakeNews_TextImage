from __future__ import annotations
import json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import Config
from model import FakeNewsClassifier


def save_predictions_json(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path
):
    """Save predictions as a JSON list of records."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for y, yhat, p in zip(labels.tolist(), predictions.tolist(), probabilities.tolist()):
        records.append({
            "true_label": int(y),
            "predicted_label": int(yhat),
            "probability_fake": float(p),
            "correct": bool(y == yhat)
        })
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] Predictions (JSON) saved to {save_path}")


class Evaluator:
    """Model Evaluator"""

    def __init__(self, model: FakeNewsClassifier, config: Config):
        self.model = model
        self.config = config
        self.device = config.model.device
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataloader: DataLoader):
        """
        Complete evaluation on dataloader
        Returns:
          (metrics dict, labels np.ndarray, preds np.ndarray, probs np.ndarray)
        """
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch["images"]
                texts = batch["texts"]
                labels = batch["labels"]

                logits = self.model(images, texts)
                probs = torch.softmax(logits, dim=1)  # [B, 2]
                _, predicted = logits.max(1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy()[:, 1])  # P(fake)

        labels_np = np.array(all_labels)
        preds_np = np.array(all_predictions)
        probs_np = np.array(all_probabilities)

        metrics = {
            "accuracy": float(accuracy_score(labels_np, preds_np)),
            "precision": float(precision_score(labels_np, preds_np)),
            "recall": float(recall_score(labels_np, preds_np)),
            "f1": float(f1_score(labels_np, preds_np)),
            "roc_auc": float(roc_auc_score(labels_np, probs_np))
        }

        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}")

        print("\nClassification Report:")
        print(classification_report(labels_np, preds_np, target_names=["Real", "Fake"]))

        cm = confusion_matrix(labels_np, preds_np)
        self.plot_confusion_matrix(cm)

        return metrics, labels_np, preds_np, probs_np

    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"]
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        save_path = self.config.log_dir / "confusion_matrix.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"\n[SAVE] Confusion matrix saved to {save_path}")
        plt.close()


def plot_training_history(train_losses, val_losses, val_accuracies, config: Config):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)

    # Val accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.grid(True)

    plt.tight_layout()
    save_path = config.log_dir / "training_history.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"[SAVE] Training history saved to {save_path}")
    plt.close()
