from __future__ import annotations
import json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import Config
from model import FakeNewsClassifier


def save_predictions_json(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
    save_path: Path,
    num_classes: int,
    class_names: list = None
):
    """Save predictions with detailed scores as a JSON list of records."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    records = []

    for i, (y, yhat) in enumerate(zip(labels.tolist(), predictions.tolist())):
        record = {
            "sample_id": i,
            "true_label": int(y),
            "true_class_name": class_names[int(y)],
            "predicted_label": int(yhat),
            "predicted_class_name": class_names[int(yhat)],
            "correct": bool(y == yhat),
            "confidence": float(probabilities[i, yhat]),  # Confidence in predicted class
            "true_class_probability": float(probabilities[i, int(y)]),  # Probability of true class
        }

        class_probs = {}
        for cls_idx in range(num_classes):
            class_probs[class_names[cls_idx]] = float(probabilities[i, cls_idx])
        record["all_probabilities"] = class_probs

        class_logits = {}
        for cls_idx in range(num_classes):
            class_logits[class_names[cls_idx]] = float(logits[i, cls_idx])
        record["all_logits"] = class_logits

        record["prediction_entropy"] = float(-np.sum(probabilities[i] * np.log(probabilities[i] + 1e-10)))
        record["max_probability"] = float(np.max(probabilities[i]))
        record["probability_margin"] = float(np.sort(probabilities[i])[-1] - np.sort(probabilities[i])[-2])

        if not record["correct"]:
            record["error_type"] = f"predicted_{class_names[yhat]}_actual_{class_names[int(y)]}"
        else:
            record["error_type"] = None

        records.append(record)

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

        # Define class names based on classification type
        self.class_names = self._get_class_names()

    def _get_class_names(self):
        """Get class names based on classification type"""
        if self.config.data.classification_type == "2_way":
            return ["Real", "Fake"]
        elif self.config.data.classification_type == "3_way":
            return ["Real", "Fake (True Text)", "Fake (False Text)"]
        elif self.config.data.classification_type == "6_way":
            return ["Real", "Satire/Parody", "Misleading", "Imposter", "False Connection", "Manipulated"]
        else:
            return [f"Class {i}" for i in range(self.config.data.num_classes)]

    def evaluate(self, dataloader: DataLoader):
        """
        Complete evaluation on dataloader
        Returns:
          (metrics dict, labels np.ndarray, preds np.ndarray, probs np.ndarray, logits np.ndarray)
        """
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch["images"]
                texts = batch["texts"]
                labels = batch["labels"]

                logits = self.model(images, texts)
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        labels_np = np.array(all_labels)
        preds_np = np.array(all_predictions)
        probs_np = np.array(all_probabilities)
        logits_np = np.array(all_logits)

        # calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(labels_np, preds_np)),
        }

        # weighted metrics for multi-class
        metrics["precision"] = float(precision_score(labels_np, preds_np, average='weighted', zero_division=0))
        metrics["recall"] = float(recall_score(labels_np, preds_np, average='weighted', zero_division=0))
        metrics["f1"] = float(f1_score(labels_np, preds_np, average='weighted', zero_division=0))

        # ROC-AUC
        try:
            if self.config.data.num_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(labels_np, probs_np[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(labels_np, probs_np, multi_class='ovr'))
        except ValueError as e:
            print(f"[WARNING] Could not calculate ROC-AUC: {e}")
            metrics["roc_auc"] = None

        metrics["average_confidence"] = float(np.mean(np.max(probs_np, axis=1)))
        metrics["average_entropy"] = float(np.mean(-np.sum(probs_np * np.log(probs_np + 1e-10), axis=1)))

        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        for k, v in metrics.items():
            if v is not None:
                print(f"{k.upper()}: {v:.4f}")

        print("\nClassification Report:")
        print(classification_report(labels_np, preds_np, target_names=self.class_names, zero_division=0))


        self.save_metrics_json(metrics, self.config.log_dir / f"metrics_{self.config.data.classification_type}.json")

        cm = confusion_matrix(labels_np, preds_np)
        self.save_confusion_matrix_data(
            cm,
            self.class_names,
            self.config.log_dir / f"confusion_matrix_{self.config.data.classification_type}"
        )
        self.plot_confusion_matrix(cm)

        self.save_roc_data(
            labels_np,
            probs_np,
            self.class_names,
            self.config.data.num_classes,
            self.config.log_dir / f"roc_data_{self.config.data.classification_type}.json"
        )
        self.plot_roc_curve(labels_np, probs_np)

        self.save_pr_data(
            labels_np,
            probs_np,
            self.class_names,
            self.config.data.num_classes,
            self.config.log_dir / f"pr_data_{self.config.data.classification_type}.json"
        )
        self.plot_precision_recall_curve(labels_np, probs_np)

        return metrics, labels_np, preds_np, probs_np, logits_np

    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        figsize = (8, 6) if self.config.data.num_classes <= 3 else (12, 10)
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
        plt.title(f"Confusion Matrix ({self.config.data.classification_type})")
        plt.tight_layout()
        save_path = self.config.log_dir / f"confusion_matrix_{self.config.data.classification_type}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[SAVE] Confusion matrix saved to {save_path}")
        plt.close()

    def plot_roc_curve(self, labels: np.ndarray, probabilities: np.ndarray):
        """Plot ROC curve(s) for the model"""
        from sklearn.metrics import roc_curve, auc

        num_classes = self.config.data.num_classes

        if num_classes == 2:
            # Binary classification - single ROC curve
            fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.config.data.classification_type}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            save_path = self.config.log_dir / f"roc_curve_{self.config.data.classification_type}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVE] ROC curve saved to {save_path}")
            plt.close()

        else:
            # multi-class - one ROC curve per class
            from sklearn.preprocessing import label_binarize

            # binarize labels for multi-class ROC
            labels_bin = label_binarize(labels, classes=range(num_classes))

            plt.figure(figsize=(10, 8))

            # plot ROC curve for each class
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                        label=f'{self.class_names[i]} (AUC = {roc_auc:.4f})')

            # plot micro-average ROC curve
            fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), probabilities.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, lw=2, linestyle=':', color='deeppink',
                    label=f'Micro-average (AUC = {roc_auc_micro:.4f})')

            # plot macro-average ROC curve
            all_fpr = np.unique(np.concatenate([roc_curve(labels_bin[:, i], probabilities[:, i])[0]
                                                for i in range(num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= num_classes
            roc_auc_macro = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, lw=2, linestyle='--', color='navy',
                    label=f'Macro-average (AUC = {roc_auc_macro:.4f})')

            # random classifier line
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {self.config.data.classification_type}')
            plt.legend(loc="lower right", fontsize=9)
            plt.grid(True, alpha=0.3)

            save_path = self.config.log_dir / f"roc_curve_{self.config.data.classification_type}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVE] ROC curves saved to {save_path}")
            plt.close()

    def plot_precision_recall_curve(self, labels: np.ndarray, probabilities: np.ndarray):
        """Plot Precision-Recall curve(s)"""
        from sklearn.metrics import precision_recall_curve, average_precision_score

        num_classes = self.config.data.num_classes

        if num_classes == 2:
            # Binary classification
            precision, recall, thresholds = precision_recall_curve(labels, probabilities[:, 1])
            avg_precision = average_precision_score(labels, probabilities[:, 1])

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AP = {avg_precision:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {self.config.data.classification_type}')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            save_path = self.config.log_dir / f"pr_curve_{self.config.data.classification_type}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVE] Precision-Recall curve saved to {save_path}")
            plt.close()

        else:
            # Multi-class
            from sklearn.preprocessing import label_binarize

            labels_bin = label_binarize(labels, classes=range(num_classes))

            plt.figure(figsize=(10, 8))

            for i in range(num_classes):
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
            plt.title(f'Precision-Recall Curves - {self.config.data.classification_type}')
            plt.legend(loc="lower left", fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            save_path = self.config.log_dir / f"pr_curve_{self.config.data.classification_type}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
        plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = self.config.log_dir / f"confidence_distribution_{self.config.data.classification_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVE] Confidence distribution saved to {save_path}")
        plt.close()

    def save_metrics_json(self, metrics: dict, save_path: Path):
        """Save evaluation metrics to JSON"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Metrics saved to {save_path}")


    def save_confusion_matrix_data(self, cm: np.ndarray, class_names: list, save_path: Path):
        """Save confusion matrix as JSON and CSV"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm_dict = {
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "normalized": False
        }

        json_path = save_path.parent / f"{save_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cm_dict, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Confusion matrix data saved to {json_path}")



    def save_roc_data(self, labels: np.ndarray, probabilities: np.ndarray,
                    class_names: list, num_classes: int, save_path: Path):
        """Save ROC curve data (FPR, TPR, thresholds, AUC)"""

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        roc_data = {}

        if num_classes == 2:
            # binary classification
            fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)

            roc_data = {
                "classification_type": "binary",
                "class_names": class_names,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
                "auc": float(roc_auc)
            }

        else:
            # multi-class
            labels_bin = label_binarize(labels, classes=range(num_classes))

            roc_data = {
                "classification_type": "multi_class",
                "class_names": class_names,
                "per_class": {}
            }

            # per-class ROC
            for i in range(num_classes):
                fpr, tpr, thresholds = roc_curve(labels_bin[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)

                roc_data["per_class"][class_names[i]] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                    "auc": float(roc_auc)
                }

            # micro-average
            fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), probabilities.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            roc_data["micro_average"] = {
                "fpr": fpr_micro.tolist(),
                "tpr": tpr_micro.tolist(),
                "auc": float(roc_auc_micro)
            }

            # macro-average
            all_fpr = np.unique(np.concatenate([roc_curve(labels_bin[:, i], probabilities[:, i])[0]
                                                for i in range(num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= num_classes
            roc_auc_macro = auc(all_fpr, mean_tpr)
            roc_data["macro_average"] = {
                "fpr": all_fpr.tolist(),
                "tpr": mean_tpr.tolist(),
                "auc": float(roc_auc_macro)
            }

        # save as JSON
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(roc_data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] ROC data saved to {save_path}")


    def save_pr_data(self, labels: np.ndarray, probabilities: np.ndarray,
                    class_names: list, num_classes: int, save_path: Path):
        """Save Precision-Recall curve data"""

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pr_data = {}

        if num_classes == 2:
            # binary classification
            precision, recall, thresholds = precision_recall_curve(labels, probabilities[:, 1])
            avg_precision = average_precision_score(labels, probabilities[:, 1])

            pr_data = {
                "classification_type": "binary",
                "class_names": class_names,
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
                "average_precision": float(avg_precision)
            }

        else:
            # multi-class
            labels_bin = label_binarize(labels, classes=range(num_classes))

            pr_data = {
                "classification_type": "multi_class",
                "class_names": class_names,
                "per_class": {}
            }

            # per-class PR
            for i in range(num_classes):
                precision, recall, thresholds = precision_recall_curve(labels_bin[:, i], probabilities[:, i])
                avg_precision = average_precision_score(labels_bin[:, i], probabilities[:, i])

                pr_data["per_class"][class_names[i]] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": thresholds.tolist(),
                    "average_precision": float(avg_precision)
                }

            # micro-average
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


def plot_training_history(train_losses, val_losses, val_accuracies, config: Config):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))

    # loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)

    # val accuracy plot
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
