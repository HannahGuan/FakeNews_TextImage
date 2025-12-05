"""
Evaluation script for OLD baseline_model.py checkpoints
Compatible with checkpoints saved by baseline_model.py (without 'config' field)

Author: Adapted for compatibility
"""

import json
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import BertTokenizer
from torchvision import transforms

from baseline_model import MultiModalDataset, MultiModalClassifier


class ModelEvaluator:
    """Evaluator for old baseline checkpoints"""

    def __init__(
        self,
        model,
        num_classes: int,
        classification_type: str,
        device: torch.device,
        output_dir: Path = Path('evaluation_results')
    ):
        self.model = model
        self.num_classes = num_classes
        self.classification_type = classification_type
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.model.eval()

        self.class_names = self._get_class_names()

    def _get_class_names(self):
        """Get class names based on classification type"""
        if self.classification_type == "2_way":
            return ["Real", "Fake"]
        elif self.classification_type == "3_way":
            return ["Fake", "Satire", "Real"]
        elif self.classification_type == "6_way":
            return ["Fake", "Satire/Parody", "Misleading", "Imposter", "False Connection", "Manipulated"]
        else:
            return [f"Class {i}" for i in range(self.num_classes)]

    def evaluate(self, dataloader: DataLoader):
        """Complete evaluation on dataloader"""
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_logits = []

        print(f"\nEvaluating model on {len(dataloader)} batches...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label']

                # Forward pass
                logits = self.model(input_ids, attention_mask, images)
                probs = torch.softmax(logits, dim=1)
                predicted = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        labels_np = np.array(all_labels)
        preds_np = np.array(all_predictions)
        probs_np = np.array(all_probabilities)
        logits_np = np.array(all_logits)

        # Calculate metrics
        metrics = self._calculate_metrics(labels_np, preds_np, probs_np)

        # Print results
        self._print_results(metrics, labels_np, preds_np)

        # Save results
        self._save_all_results(metrics, labels_np, preds_np, probs_np, logits_np)

        return metrics, labels_np, preds_np, probs_np, logits_np

    def _calculate_metrics(self, labels, predictions, probabilities):
        """Calculate all evaluation metrics"""
        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, average='weighted', zero_division=0)),
            "recall": float(recall_score(labels, predictions, average='weighted', zero_division=0)),
            "f1": float(f1_score(labels, predictions, average='weighted', zero_division=0)),
        }

        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)

        metrics["per_class"] = {}
        for i, class_name in enumerate(self.class_names):
            metrics["per_class"][class_name] = {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i])
            }

        # ROC-AUC
        try:
            if self.num_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(labels, probabilities, multi_class='ovr'))
        except ValueError as e:
            print(f"[WARNING] Could not calculate ROC-AUC: {e}")
            metrics["roc_auc"] = None

        # Confidence and entropy metrics
        metrics["average_confidence"] = float(np.mean(np.max(probabilities, axis=1)))
        metrics["average_entropy"] = float(np.mean(-np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)))

        return metrics

    def _print_results(self, metrics, labels, predictions):
        """Print evaluation results to console"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Classification Type: {self.classification_type}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Total Samples: {len(labels)}")
        print("-"*70)

        for k, v in metrics.items():
            if k != "per_class" and v is not None:
                print(f"{k.upper()}: {v:.4f}")

        print("\n" + "-"*70)
        print("Per-Class Metrics:")
        print("-"*70)
        for class_name, class_metrics in metrics["per_class"].items():
            print(f"\n{class_name}:")
            for metric_name, value in class_metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        print("\n" + "-"*70)
        print("Classification Report:")
        print("-"*70)
        print(classification_report(labels, predictions, target_names=self.class_names, zero_division=0))

    def _save_all_results(self, metrics, labels, predictions, probabilities, logits):
        """Save all evaluation results"""
        # Save metrics JSON
        self.save_metrics_json(
            metrics,
            self.output_dir / f"metrics_{self.classification_type}.json"
        )

        # Save predictions JSON
        self.save_predictions_json(
            labels, predictions, probabilities, logits,
            self.output_dir / f"predictions_{self.classification_type}.json"
        )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        self.save_confusion_matrix_data(
            cm,
            self.output_dir / f"confusion_matrix_{self.classification_type}.json"
        )
        self.plot_confusion_matrix(cm)

        # ROC curves
        self.save_roc_data(
            labels, probabilities,
            self.output_dir / f"roc_data_{self.classification_type}.json"
        )
        self.plot_roc_curve(labels, probabilities)

        # Precision-Recall curves
        self.save_pr_data(
            labels, probabilities,
            self.output_dir / f"pr_data_{self.classification_type}.json"
        )
        self.plot_precision_recall_curve(labels, probabilities)

        # Confidence distribution
        self.plot_confidence_distribution(probabilities, predictions, labels)

    def save_metrics_json(self, metrics: dict, save_path: Path):
        """Save evaluation metrics to JSON"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n[SAVE] Metrics saved to {save_path}")

    def save_predictions_json(self, labels, predictions, probabilities, logits, save_path: Path):
        """Save detailed predictions with scores"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for i in range(len(labels)):
            y_true = int(labels[i])
            y_pred = int(predictions[i])

            record = {
                "sample_id": i,
                "true_label": y_true,
                "true_class_name": self.class_names[y_true],
                "predicted_label": y_pred,
                "predicted_class_name": self.class_names[y_pred],
                "correct": bool(y_true == y_pred),
                "confidence": float(probabilities[i, y_pred]),
                "true_class_probability": float(probabilities[i, y_true]),
            }

            # All class probabilities
            class_probs = {}
            for cls_idx, class_name in enumerate(self.class_names):
                class_probs[class_name] = float(probabilities[i, cls_idx])
            record["all_probabilities"] = class_probs

            # All class logits
            class_logits = {}
            for cls_idx, class_name in enumerate(self.class_names):
                class_logits[class_name] = float(logits[i, cls_idx])
            record["all_logits"] = class_logits

            # Uncertainty metrics
            record["prediction_entropy"] = float(-np.sum(probabilities[i] * np.log(probabilities[i] + 1e-10)))
            record["max_probability"] = float(np.max(probabilities[i]))
            sorted_probs = np.sort(probabilities[i])
            record["probability_margin"] = float(sorted_probs[-1] - sorted_probs[-2])

            # Error type
            if not record["correct"]:
                record["error_type"] = f"predicted_{self.class_names[y_pred]}_actual_{self.class_names[y_true]}"
            else:
                record["error_type"] = None

            records.append(record)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Predictions saved to {save_path}")

    def save_confusion_matrix_data(self, cm: np.ndarray, save_path: Path):
        """Save confusion matrix as JSON"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm_dict = {
            "confusion_matrix": cm.tolist(),
            "class_names": self.class_names,
            "normalized": False
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(cm_dict, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Confusion matrix data saved to {save_path}")

    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        figsize = (8, 6) if self.num_classes <= 3 else (12, 10)
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title(f"Confusion Matrix ({self.classification_type})")
        plt.tight_layout()

        save_path = self.output_dir / f"confusion_matrix_{self.classification_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVE] Confusion matrix plot saved to {save_path}")
        plt.close()

    def save_roc_data(self, labels: np.ndarray, probabilities: np.ndarray, save_path: Path):
        """Save ROC curve data"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        roc_data = {}

        if self.num_classes == 2:
            # Binary classification
            fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
            roc_auc_value = auc(fpr, tpr)

            roc_data = {
                "classification_type": "binary",
                "class_names": self.class_names,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
                "auc": float(roc_auc_value)
            }
        else:
            # Multi-class
            labels_bin = label_binarize(labels, classes=range(self.num_classes))

            roc_data = {
                "classification_type": "multi_class",
                "class_names": self.class_names,
                "per_class": {}
            }

            # Per-class ROC
            for i in range(self.num_classes):
                fpr, tpr, thresholds = roc_curve(labels_bin[:, i], probabilities[:, i])
                roc_auc_value = auc(fpr, tpr)

                roc_data["per_class"][self.class_names[i]] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                    "auc": float(roc_auc_value)
                }

            # Micro-average
            fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), probabilities.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            roc_data["micro_average"] = {
                "fpr": fpr_micro.tolist(),
                "tpr": tpr_micro.tolist(),
                "auc": float(roc_auc_micro)
            }

            # Macro-average
            all_fpr = np.unique(np.concatenate([roc_curve(labels_bin[:, i], probabilities[:, i])[0]
                                                for i in range(self.num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= self.num_classes
            roc_auc_macro = auc(all_fpr, mean_tpr)
            roc_data["macro_average"] = {
                "fpr": all_fpr.tolist(),
                "tpr": mean_tpr.tolist(),
                "auc": float(roc_auc_macro)
            }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(roc_data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] ROC data saved to {save_path}")

    def plot_roc_curve(self, labels: np.ndarray, probabilities: np.ndarray):
        """Plot ROC curve(s)"""
        if self.num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            roc_auc_value = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc_value:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.classification_type}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            save_path = self.output_dir / f"roc_curve_{self.classification_type}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVE] ROC curve saved to {save_path}")
            plt.close()

        else:
            # Multi-class
            labels_bin = label_binarize(labels, classes=range(self.num_classes))

            plt.figure(figsize=(10, 8))

            # Plot ROC curve for each class
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
                roc_auc_value = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                        label=f'{self.class_names[i]} (AUC = {roc_auc_value:.4f})')

            # Micro-average
            fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), probabilities.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, lw=2, linestyle=':', color='deeppink',
                    label=f'Micro-average (AUC = {roc_auc_micro:.4f})')

            # Macro-average
            all_fpr = np.unique(np.concatenate([roc_curve(labels_bin[:, i], probabilities[:, i])[0]
                                                for i in range(self.num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= self.num_classes
            roc_auc_macro = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, lw=2, linestyle='--', color='navy',
                    label=f'Macro-average (AUC = {roc_auc_macro:.4f})')

            # Random classifier line
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {self.classification_type}')
            plt.legend(loc="lower right", fontsize=9)
            plt.grid(True, alpha=0.3)

            save_path = self.output_dir / f"roc_curve_{self.classification_type}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVE] ROC curves saved to {save_path}")
            plt.close()

    def save_pr_data(self, labels: np.ndarray, probabilities: np.ndarray, save_path: Path):
        """Save Precision-Recall curve data"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pr_data = {}

        if self.num_classes == 2:
            # Binary classification
            precision, recall, thresholds = precision_recall_curve(labels, probabilities[:, 1])
            avg_precision = average_precision_score(labels, probabilities[:, 1])

            pr_data = {
                "classification_type": "binary",
                "class_names": self.class_names,
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
                "average_precision": float(avg_precision)
            }
        else:
            # Multi-class
            labels_bin = label_binarize(labels, classes=range(self.num_classes))

            pr_data = {
                "classification_type": "multi_class",
                "class_names": self.class_names,
                "per_class": {}
            }

            # Per-class PR
            for i in range(self.num_classes):
                precision, recall, thresholds = precision_recall_curve(labels_bin[:, i], probabilities[:, i])
                avg_precision = average_precision_score(labels_bin[:, i], probabilities[:, i])

                pr_data["per_class"][self.class_names[i]] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": thresholds.tolist(),
                    "average_precision": float(avg_precision)
                }

            # Micro-average
            precision_micro, recall_micro, _ = precision_recall_curve(
                labels_bin.ravel(), probabilities.ravel())
            avg_precision_micro = average_precision_score(labels_bin, probabilities, average='micro')
            pr_data["micro_average"] = {
                "precision": precision_micro.tolist(),
                "recall": recall_micro.tolist(),
                "average_precision": float(avg_precision_micro)
            }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(pr_data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Precision-Recall data saved to {save_path}")

    def plot_precision_recall_curve(self, labels: np.ndarray, probabilities: np.ndarray):
        """Plot Precision-Recall curve(s)"""
        if self.num_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
            avg_precision = average_precision_score(labels, probabilities[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AP = {avg_precision:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {self.classification_type}')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            save_path = self.output_dir / f"pr_curve_{self.classification_type}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVE] Precision-Recall curve saved to {save_path}")
            plt.close()

        else:
            # Multi-class
            labels_bin = label_binarize(labels, classes=range(self.num_classes))

            plt.figure(figsize=(10, 8))

            for i in range(self.num_classes):
                precision, recall, _ = precision_recall_curve(labels_bin[:, i], probabilities[:, i])
                avg_precision = average_precision_score(labels_bin[:, i], probabilities[:, i])
                plt.plot(recall, precision, lw=2,
                        label=f'{self.class_names[i]} (AP = {avg_precision:.4f})')

            # Micro-average
            precision_micro, recall_micro, _ = precision_recall_curve(
                labels_bin.ravel(), probabilities.ravel())
            avg_precision_micro = average_precision_score(labels_bin, probabilities, average='micro')
            plt.plot(recall_micro, precision_micro, lw=2, linestyle=':', color='deeppink',
                    label=f'Micro-average (AP = {avg_precision_micro:.4f})')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves - {self.classification_type}')
            plt.legend(loc="lower left", fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            save_path = self.output_dir / f"pr_curve_{self.classification_type}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVE] Precision-Recall curves saved to {save_path}")
            plt.close()

    def plot_confidence_distribution(self, probs: np.ndarray, preds: np.ndarray, labels: np.ndarray):
        """Plot confidence distribution for correct vs incorrect predictions"""
        confidences = np.max(probs, axis=1)
        correct_mask = (preds == labels)

        plt.figure(figsize=(10, 6))
        plt.hist(confidences[correct_mask], bins=50, alpha=0.7, label='Correct', color='green')
        plt.hist(confidences[~correct_mask], bins=50, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence (Max Probability)')
        plt.ylabel('Count')
        plt.title(f'Confidence Distribution: Correct vs Incorrect Predictions ({self.classification_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = self.output_dir / f"confidence_distribution_{self.classification_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVE] Confidence distribution saved to {save_path}")
        plt.close()


def main(args):
    """Main evaluation function"""

    # Configuration
    LABEL_TYPE = args.classification_type
    NUM_CLASSES = {'2_way': 2, '3_way': 3, '6_way': 6}[LABEL_TYPE]
    MAX_LENGTH = args.max_length
    BATCH_SIZE = args.batch_size

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Using device: {device}")
    print(f"Classification type: {LABEL_TYPE} ({NUM_CLASSES} classes)")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    # Initialize model (using default params from baseline_model.py)
    model = MultiModalClassifier(
        num_classes=NUM_CLASSES,
        hidden_dim=512,
        dropout=0.3
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Image transformations
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    print(f"\nLoading test dataset from {args.test_csv}")
    test_dataset = MultiModalDataset(
        str(args.test_csv),
        str(args.test_images_dir),
        tokenizer,
        image_transform,
        max_length=MAX_LENGTH,
        label_type=LABEL_TYPE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        num_classes=NUM_CLASSES,
        classification_type=LABEL_TYPE,
        device=device,
        output_dir=Path(args.output_dir)
    )

    # Run evaluation
    metrics, labels, predictions, probabilities, logits = evaluator.evaluate(test_loader)

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate OLD Baseline Checkpoints (from baseline_model.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default='checkpoints/2_way/best_model.pth',
        help='Path to old model checkpoint'
    )
    parser.add_argument(
        '--classification-type',
        type=str,
        default='2_way',
        choices=['2_way', '3_way', '6_way'],
        help='Classification type'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default='../Data/test_split_2_way.csv',
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--test-images-dir',
        type=str,
        default='../Data/dev_images',
        help='Path to test images directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Maximum sequence length for tokenization'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )

    args = parser.parse_args()
    main(args)
